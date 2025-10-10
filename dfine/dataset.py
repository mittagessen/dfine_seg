#
# Copyright 2025 Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
import cv2
import torch
import random
import logging
import numpy as np
import lightning as L
import albumentations as A

from kraken.lib.xml import XMLPage
from collections import defaultdict

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from typing import Optional, Union, TYPE_CHECKING
from collections.abc import Iterable, Sequence

from albumentations.pytorch.transforms import ToTensorV2

from dfine.configs import AUGMENTATION_CONFIG

if TYPE_CHECKING:
    from os import PathLike

logger = logging.getLogger(__name__)


def filter_bboxes(bboxes: torch.Tensor) -> torch.Tensor:
    """
    Filters bounding boxes in normalized cxcywh format to be inside the image and be
    ordered correctly.

    Args:
        bboxes: tensor of shape `(T, 4)` containing the bbox coordinates
        (center_x, center_y, width, height).
        image_size: tuple with the input image size (width, height).

    Returns:
        A tensors of shape `(T', 4)` where `T' <= T`.
    """
    close_to_zero = bboxes.isclose(torch.tensor([0.]))
    close_to_one = bboxes.isclose(torch.tensor([1.]))

    valid_range = ~(close_to_zero | close_to_one)

    valid_width = (bboxes[:, 0] - bboxes[:, 2] / 2 > 0) & (bboxes[:, 0] + bboxes[:, 2] / 2 < 1)
    valid_height = (bboxes[:, 1] - bboxes[:, 3] / 2 > 0) & (bboxes[:, 1] + bboxes[:, 3] / 2 < 1)
    valid_bboxes = valid_range & valid_width & valid_height
    return bboxes[valid_bboxes]


def collate_batch(batch):
    """
    Collates an object detection batch.
    """
    images = torch.stack([sample.pop('image') for sample in batch])
    return {'images': images,
            'target': batch}


def polygon_to_cxcywh(polygon: Iterable[tuple[int, int]], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """
    Computes the minimal bounding box in xyxy format of a polygon and return it
    in normalized cxcyhw form.

    Args:
        boxes: An iterable of tuples of the format (x0, y0), (x1,
                          y1), ... (xn, yn).
        image_size: A tuple (x_max, y_max).

    Returns:
        A normalized box (center_x, center_y, width, height) covering all
        points in the input polygon.
    """
    w, h = image_size
    flat_box = [point for pol in polygon for point in pol]
    xmin, xmax = min(flat_box[::2]), max(flat_box[::2])
    ymin, ymax = min(flat_box[1::2]), max(flat_box[1::2])
    cx = (xmin + xmax) / 2 / w
    cy = (ymin + ymax) / 2 / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    return cx, cy, bw, bh


class XMLDetectionDataset(Dataset):
    """
    An object detection dataset parsing regions in ALTO or PageXML files.

    Output bbox are in xyxy format.

    Args:
        valid_regions:
        merge_regions:
        merge_all_regions:
        class_mapping:
    """
    def __init__(self,
                 valid_regions: Optional[Sequence[str]] = None,
                 merge_regions: Optional[dict[str, Sequence[str]]] = None,
                 merge_all_regions: Optional[str] = None,
                 class_mapping: Optional[dict[str, int]] = None,
                 augmentation: bool = False,
                 augmentation_config: dict = AUGMENTATION_CONFIG,
                 image_size: tuple[int, int] = (1280, 1280)):
        super().__init__()
        if class_mapping:
            self.class_mapping = class_mapping
            self.num_classes = len(self.class_mapping)
            self.freeze_cls_map = True
        else:
            self.class_mapping = {}
            self.num_classes = 0
            self.freeze_cls_map = False

        self.class_stats = defaultdict(int)

        self.mreg_dict = merge_regions if merge_regions is not None else {}
        self.valid_regions = valid_regions
        self.merge_all_regions = merge_all_regions

        self.image_size = image_size
        self.norm = ([0, 0, 0], [1, 1, 1])

        self.augmentation = augmentation

        cv2.setNumThreads(0)
        if augmentation:
            self.mosaic_prob = augmentation_config['mosaic_prob']
            self.mosaic_transform = A.Compose([A.Mosaic(grid_yx=(3, 3),
                                                        target_size=image_size),
                                               A.Normalize(mean=self.norm[0], std=self.norm[1]),
                                               ToTensorV2()],
                                              bbox_params=A.BboxParams(format="yolo", label_fields=["labels"]))

            self.transforms = A.Compose([A.CoarseDropout(
                                             num_holes_range=(1, 2),
                                             hole_height_range=(0.05, 0.15),
                                             hole_width_range=(0.05, 0.15),
                                             p=augmentation_config['coarse_dropout'],
                                         ),
                                         A.RandomBrightnessContrast(p=augmentation_config['brightness']),
                                         A.RandomGamma(p=augmentation_config['gamma']),
                                         A.Blur(p=augmentation_config['blur']),
                                         A.GaussNoise(p=augmentation_config['noise'], std_range=(0.1, 0.2)),
                                         A.ToGray(p=augmentation_config['to_gray']),
                                         A.Affine(
                                             rotate=[90, 90],
                                             p=augmentation_config['rotate_90'],
                                             fit_output=True,
                                         ),
                                         A.HorizontalFlip(p=augmentation_config['left_right_flip']),
                                         A.VerticalFlip(p=augmentation_config['up_down_flip']),
                                         A.Rotate(
                                             limit=augmentation_config['rotation_degree'],
                                             p=augmentation_config['rotation_p'],
                                             interpolation=cv2.INTER_AREA,
                                             border_mode=cv2.BORDER_CONSTANT,
                                             fill=(114, 114, 114),
                                         ),
                                         A.Resize(self.image_size[0], self.image_size[1], interpolation=cv2.INTER_AREA),
                                         A.Normalize(mean=self.norm[0], std=self.norm[1]),
                                         ToTensorV2()],
                                        bbox_params=A.BboxParams(format="yolo", label_fields=["labels"]))
        else:
            self.transforms = A.Compose([A.Resize(self.image_size[0], self.image_size[1], interpolation=cv2.INTER_AREA),
                                         A.Normalize(mean=self.norm[0], std=self.norm[1]),
                                         ToTensorV2()],
                                        bbox_params=A.BboxParams(format="yolo", label_fields=["labels"]))

        self.targets = []
        self.imgs = []

    def add(self, doc: 'XMLPage'):
        """
        Adds a page to the dataset.

        Args:
            doc: An XMLPage parser class.
        """
        regions_ = defaultdict(list)
        wh = doc.image_size
        for k, v in doc._regions.items():
            v = torch.Tensor([polygon_to_cxcywh(x.boundary, wh) for x in v if x.boundary])
            # v = filter_bboxes(torch.Tensor(v))

            if self.valid_regions is None or k in self.valid_regions:
                reg_type = self.mreg_dict.get(k, k)
                if self.merge_all_regions:
                    reg_type = self.merge_all_regions
                if reg_type not in self.class_mapping and self.freeze_cls_map:
                    continue
                elif reg_type not in self.class_mapping:
                    self.num_classes += 1
                    self.class_mapping[reg_type] = self.num_classes - 1
                regions_[reg_type].extend(v)
                self.class_stats[reg_type] += len(v)
        self.targets.append(dict(regions_))
        self.imgs.append(doc.imagename)

    def _get_sample(self, idx):
        image = cv2.imread(self.imgs[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        labels = []
        bboxes = []

        for k, v in self.targets[idx].items():
            labels.extend(len(v) * [self.class_mapping[k]])
            bboxes.extend(v)

        return {'image': image,
                'labels': labels,
                'bboxes': bboxes}

    def __getitem__(self, idx):
        if self.augmentation and random.random() < self.mosaic_prob:
            res = self._get_sample(idx)
            # get additional samples for composing the mosaic
            add_samples = [self._get_sample(random.randint(0, len(self) - 1)) for _ in range(3)]
            res = self.mosaic_transform(**self._get_sample(idx), mosaic_metadata=add_samples)
        else:
            res = self._get_sample(idx)
            res = self.transforms(image=res['image'], labels=np.array(res['labels']), bboxes=np.array(res['bboxes']))
        return {'image': res['image'],
                'labels': torch.tensor(res['labels'], dtype=torch.int64),
                'boxes': torch.tensor(res['bboxes'], dtype=torch.float)}

    def __len__(self):
        return len(self.imgs)

    def close_mosaic(self):
        self.mosaic_prob = 0.0


class RegionDetectionDataModule(L.LightningDataModule):
    def __init__(self,
                 training_data: list[Union[str, 'PathLike']],
                 evaluation_data: list[Union[str, 'PathLike']],
                 valid_regions: Sequence[str] = None,
                 merge_regions: dict[str, Sequence[str]] = None,
                 merge_all_regions: Optional[str] = None,
                 class_mapping: Optional[dict[str, int]] = None,
                 image_size: tuple[int, int] = (320, 320),
                 batch_size: int = 16,
                 num_workers: int = 8,
                 **kwargs):
        super().__init__()

        self.save_hyperparameters()
        self.train_set = XMLDetectionDataset(valid_regions=valid_regions,
                                             merge_regions=merge_regions,
                                             merge_all_regions=merge_all_regions,
                                             augmentation=False,
                                             class_mapping=class_mapping,
                                             image_size=image_size)

        for file in training_data:
            try:
                self.train_set.add(XMLPage(file))
            except Exception as e:
                logger.warning(f'Failed to parse {file}: {e}')

        self.val_set = XMLDetectionDataset(image_size=image_size,
                                           augmentation=False,
                                           class_mapping=self.train_set.class_mapping)

        logger.info(f'Parsing {len(evaluation_data)} XML files for validation data')
        for file in evaluation_data:
            try:
                self.val_set.add(XMLPage(file))
            except Exception as e:
                logger.warning(f'Failed to parse {file}: {e}')

        self.num_classes = self.train_set.num_classes
        self.class_mapping = self.train_set.class_mapping

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          shuffle=True,
                          collate_fn=collate_batch)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          shuffle=False,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          collate_fn=collate_batch)
