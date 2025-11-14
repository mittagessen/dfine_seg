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
import albumentations as A

from PIL import Image
from typing import TYPE_CHECKING
from collections import defaultdict

from kraken.lib.dataset.utils import _get_type

from torch.utils.data import Dataset
from collections.abc import Iterable
from albumentations.pytorch.transforms import ToTensorV2

from dfine.configs import AUGMENTATION_CONFIG

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from kraken.containers import Segmentation


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
        class_mapping: dictionary with two
    """
    def __init__(self,
                 class_mapping: dict[str, dict[str, int]],
                 augmentation: bool = False,
                 augmentation_config: dict = AUGMENTATION_CONFIG,
                 image_size: tuple[int, int] = (1280, 1280)):
        super().__init__()
        self.class_mapping = class_mapping
        self.num_classes = max(max(v.values()) if v else 0 for v in self.class_mapping.values())

        self.failed_samples = set()
        self.class_stats = {'lines': defaultdict(int), 'regions': defaultdict(int)}

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
                                              bbox_params=A.BboxParams(format="yolo", label_fields=["labels"], clip=True, filter_invalid_bboxes=True))

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
                                        bbox_params=A.BboxParams(format="yolo", label_fields=["labels"], clip=True, filter_invalid_bboxes=True))
        else:
            self.transforms = A.Compose([A.Resize(self.image_size[0], self.image_size[1], interpolation=cv2.INTER_AREA),
                                         A.Normalize(mean=self.norm[0], std=self.norm[1]),
                                         ToTensorV2()],
                                        bbox_params=A.BboxParams(format="yolo", label_fields=["labels"], clip=True, filter_invalid_bboxes=True))

        self.targets = []
        self.imgs = []

    def add(self, doc: 'Segmentation'):
        """
        Adds a page to the dataset.

        Args:
            doc: a Segmentation object.
        """
        wh = Image.open(doc.imagename).size

        objs = defaultdict(list)
        for line in doc.lines:
            tag = _get_type(line.tags)
            try:
                idx = self.class_mapping['lines'][tag]
                objs[idx].append(polygon_to_cxcywh(line.boundary, wh))
                self.class_stats['lines'][idx] += 1
            except KeyError:
                continue

        for k, v in doc.regions.items():
            try:
                idx = self.class_mapping['regions'][k]
                v = torch.Tensor([polygon_to_cxcywh(x.boundary, wh) for x in v if x.boundary])
                objs[idx].extend(v)
                self.class_stats['regions'][idx] += len(v)
            except KeyError:
                continue
        self.targets.append(objs)
        self.imgs.append(doc.imagename)
        self.num_classes = max(max(v.values()) if v else 0 for v in self.class_mapping.values())

    def _get_sample(self, idx):
        image = cv2.imread(self.imgs[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        labels = []
        bboxes = []

        for k, v in self.targets[idx].items():
            labels.extend(len(v) * [k])
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
