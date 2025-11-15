import uuid
import torch
import torch.nn as nn

from shapely import geom
from lightning.fabric import Fabric
from collections import defaultdict
from torchvision.transforms import v2
from typing import Any, TYPE_CHECKING
from torchvision.ops import box_convert

from kraken.models import BaseModel
from kraken.configs import SegmentationInferenceConfig
from kraken.containers import Segmentation, Region, BBoxLine

from dfine.configs import models
from dfine.modules import HGNetv2, DFINETransformer, HybridEncoder

if TYPE_CHECKING:
    from PIL import Image


class DFINE(nn.Module, BaseModel):

    user_metadata = {}
    model_type = 'segmentation'
    _kraken_min_version = '7.0.0'

    def __init__(self, **kwargs):
        """
        A D-FINE object detection model.

        Args:
            model_variant: Literal['nano', 'small', 'medium', 'large', 'extra_large'],
            class_mapping: dict[str, dict[str, int]],
            image_size: tuple[int, int]
        """
        super().__init__()

        if (model_variant := kwargs.get('model_variant', None)) is None:
            raise ValueError('model_variant argument is missing in args.')
        if (class_mapping := kwargs.get('class_mapping', None)) is None:
            raise ValueError('class_mapping argument is missing in args.')
        if (image_size := kwargs.get('image_size', None)) is None:
            raise ValueError('image_size argument is missing in args.')
        if kwargs.get('num_top_queries', None) is None:
            raise ValueError('num_top_queries argument is missing in args.')

        self.user_metadata: dict[str, Any] = {'accuracy': [],
                                              'metrics': []}
        self.user_metadata.update(kwargs)

        model_cfg = models[model_variant]
        model_cfg["HybridEncoder"]["eval_spatial_size"] = image_size
        model_cfg["DFINETransformer"]["eval_spatial_size"] = image_size

        self.num_classes = max(max(v.values()) if v else 0 for v in class_mapping.values()) + 1
        # invert class_mapping
        self.line_map = {}
        self.region_map = {}
        for cls, idx in self.user_metadata['class_mapping']['lines'].items():
            # there might be multiple classes mapping to the same index -> pick the first one.
            self.line_map.setdefault(cls, idx)
        for cls, idx in self.user_metadata['class_mapping']['regions'].items():
            self.region_map.setdefault(cls, idx)

        self.backbone = HGNetv2(**model_cfg["HGNetv2"])
        self.encoder = HybridEncoder(**model_cfg["HybridEncoder"])
        self.decoder = DFINETransformer(num_classes=self.num_classes, **model_cfg["DFINETransformer"])

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)
        return x

    def prepare_for_inference(self, config: SegmentationInferenceConfig):
        """
        Configures the model for inference.
        """
        self.eval()
        self._inf_config = config

        self._fabric = Fabric(accelerator=self._inf_config.accelerator,
                              devices=self._inf_config.device,
                              precision=self._inf_config.precision)

        self.nn = self._fabric._precision.convert_module(self.nn)
        self.nn = self._fabric.to_device(self.nn)

        _m_dtype = next(self.parameters()).dtype

        self.transforms = v2.Compose(v2.Resize(self.user_metadata['image_size']),
                                     v2.RGB(),
                                     v2.ToDtype(_m_dtype, scale=True),
                                     v2.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]))

    @torch.inference_mode()
    def predict(self, im: 'Image.Image') -> Segmentation:
        """
        Runs prediction with the model.

        Args:
            im: Input image.
        """
        orig_size = self._fabric.to_device(torch.tensor(tuple(im.size * 2)))
        scaled_im = self.transforms(im)
        outputs = self(scaled_im)
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        boxes = box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxyxyxy') * orig_size

        # scores from distribution
        scores = logits.sigmoid()
        scores, index = torch.topk(scores.flatten(1), self.user_metadata['num_top_queries'], dim=-1)
        labels = index - index // self.num_classes * self.num_classes
        index = index // self.num_classes
        boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))
        # threshold by class score
        mask = scores[labels] >= 0.5
        scores, boxes, labels = scores[mask].cpu(), boxes[mask].cpu(), labels[mask].cpu()

        regions = defaultdict(list)
        _shp_regs = {}
        for score, box, label in zip(scores, boxes, labels):
            if label in self.region_map:
                region = Region(id=f'_{uuid.uuid4()}', boundary=box.tolist(), tags={'type': [{'type': self.region_map[label]}]})
                regions[self.region_map[label]].append(region)
                _shp_regs[region.id] = geom.Polygon(region.boundary)

        lines = []
        for score, box, label in zip(scores, boxes, labels):
            if label in self.line_map:
                line_ls = geom.Polygon(box.tolist()).centroid
                regs = []
                for reg_id, reg in _shp_regs.items():
                    if reg.contains(line_ls):
                        regs.append(reg_id)
                lines.append(BBoxLine(id=f'_{uuid.uuid4()}',
                                      bbox=box.tolist(),
                                      tags={'type': [{'type': self.line_map[label]}]},
                                      regions=regs))

        return Segmentation(text_direction=self._inf_config.text_direction,
                            imagename=getattr(im, 'filename', None),
                            type='bbox',
                            lines=lines,
                            regions=regions,
                            script_detection=False,
                            line_orders=[])
