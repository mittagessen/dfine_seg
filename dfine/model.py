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
"""
Lightning wrapper for D-FINE object detection training.
"""
import torch
import logging
import lightning.pytorch as L

from typing import TYPE_CHECKING, Optional, Union
from torch.optim import lr_scheduler
from torchvision.ops import box_convert
from lightning.pytorch.callbacks import EarlyStopping
from torchmetrics.detection import MeanAveragePrecision
from torch.utils.data import DataLoader, Subset, random_split

from kraken.lib.xml import XMLPage
from kraken.models import create_model
from kraken.train.utils import configure_optimizer_and_lr_scheduler

from dfine.modules import build_criterion
from dfine.dataset import XMLDetectionDataset, collate_batch
from dfine.configs import (DFINESegmentationTrainingDataConfig,
                           DFINESegmentationTrainingConfig)

if TYPE_CHECKING:
    from os import PathLike
    from kraken.models import BaseModel
    from kraken.containers import Segmentation

logger = logging.getLogger(__name__)


__all__ = ['DFINESegmentationDataModule', 'DFINESegmentationModel']


@torch.compile()
def model_step(model, criterion, batch):
    o = model(batch['images'], targets=batch['target'])
    return criterion(outputs=o, targets=batch['target'])


class DFINESegmentationDataModule(L.LightningDataModule):
    def __init__(self,
                 data_config: DFINESegmentationTrainingDataConfig):
        super().__init__()
        self.save_hyperparameters()

        all_files = [getattr(data_config, x) for x in ['training_data', 'evaluation_data', 'test_data']]

        if data_config.format_type in ['xml', 'page', 'alto']:

            def _parse_xml_set(ds_type, dataset) -> list[dict[str, 'Segmentation']]:
                if not dataset:
                    return None
                logger.info(f'Parsing {len(dataset) if dataset else 0} XML files for {ds_type} data')
                data = []
                for pos, file in enumerate(dataset):
                    try:
                        data.append(XMLPage(file, filetype=data_config.format_type).to_container())
                    except Exception as e:
                        logger.warning(f'Failed to parse {file}: {e}')
                return data

            training_data = _parse_xml_set('training', all_files[0])
            evaluation_data = _parse_xml_set('evaluation', all_files[1])
            self.test_data = _parse_xml_set('test', all_files[2])
        elif data_config.format_type is None:
            training_data = data_config.training_data
            logger.info(f'Using {len(training_data) if training_data else 0} Segmentation objects for training data')
            evaluation_data = data_config.evaluation_data
            logger.info(f'Using {len(evaluation_data) if evaluation_data else 0} Segmentation objects for evaluation data')
            self.test_data = data_config.test_data
            logger.info(f'Using {len(self.test_data) if data_config.test_data else 0} Segmentation objects for test data')
        else:
            raise ValueError(f'format_type {data_config.format_type} not in [alto, page, xml, None].')

        if training_data and evaluation_data:
            train_set = self._build_dataset(training_data,
                                            augmentation=data_config.augment,
                                            image_size=data_config.image_size,
                                            class_mapping={'lines': self.hparams.data_config.line_class_mapping,
                                                           'regions': self.hparams.data_config.region_class_mapping})
            self.train_set = Subset(train_set, range(len(train_set)))
            val_set = self._build_dataset(evaluation_data,
                                          image_size=data_config.image_size,
                                          class_mapping={'lines': self.hparams.data_config.line_class_mapping,
                                                         'regions': self.hparams.data_config.region_class_mapping})

            self.val_set = Subset(val_set, range(len(val_set)))
        elif training_data:
            train_set = self._build_dataset(training_data,
                                            augmentation=data_config.augment,
                                            image_size=data_config.image_size,
                                            class_mapping={'lines': self.hparams.data_config.line_class_mapping,
                                                           'regions': self.hparams.data_config.region_class_mapping})

            train_len = int(len(train_set)*data_config.partition)
            val_len = len(train_set) - train_len
            logger.info(f'No explicit validation data provided. Splitting off '
                        f'{val_len} (of {len(train_set)}) samples to validation '
                        'set.')
            self.train_set, self.val_set = random_split(train_set, (train_len, val_len))
        elif self.test_data:
            pass
        else:
            raise ValueError('Invalid specification of training/evaluation/test data.')

    def _build_dataset(self,
                       data,
                       **kwargs):
        dataset = XMLDetectionDataset(**kwargs)

        for page in data:
            dataset.add(page)

        return dataset

    def setup(self, stage: str = None):
        if stage == 'fit' or stage is None:
            if len(self.train_set) == 0:
                raise ValueError('No valid training data provided. Please add some.')
            if len(self.val_set) == 0:
                raise ValueError('No valid validation data provided. Please add some.')
            self.hparams.data_config.line_class_mapping = dict(self.train_set.dataset.class_mapping['lines'])
            self.hparams.data_config.region_class_mapping = dict(self.train_set.dataset.class_mapping['regions'])
            self.num_classes = max(max(v.values()) if v else 0 for v in self.train_set.dataset.class_mapping.values()) + 1
        elif stage == 'test':
            if len(self.test_data) == 0:
                raise ValueError('No valid test data provided. Please add some.')
            test_set = self._build_dataset(self.test_data,
                                           image_size=self.hparams.data_config.image_size,
                                           class_mapping=self.trainer.lightning_module.net.user_metadata['class_mapping'])
            self.test_set = Subset(test_set, range(len(test_set)))
            if len(self.test_set) == 0:
                raise ValueError('No valid test data provided. Please add some.')

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.hparams.data_config.batch_size,
                          num_workers=self.hparams.data_config.num_workers,
                          shuffle=True,
                          pin_memory=True,
                          collate_fn=collate_batch)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          shuffle=False,
                          batch_size=self.hparams.data_config.batch_size,
                          num_workers=self.hparams.data_config.num_workers,
                          collate_fn=collate_batch)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          shuffle=False,
                          batch_size=self.hparams.data_config.batch_size,
                          num_workers=self.hparams.data_config.num_workers,
                          collate_fn=collate_batch)


class DFINESegmentationModel(L.LightningModule):
    """
    A LightningModule encapsulating the training setup for a region object
    detection model.
    """
    def __init__(self,
                 config: DFINESegmentationTrainingConfig,
                 model: Optional['BaseModel'] = None):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        if model:
            self.net = model

            if self.net.model_type not in [None, 'segmentation']:
                raise ValueError(f'Model {model} is of type {self.net.model_type} while `segmentation` is expected.')
        else:
            self.net = None

        self.map = MeanAveragePrecision(box_format='xyxy',
                                        iou_type='bbox',
                                        extended_summary=True)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        loss = model_step(self.net, self.criterion, batch)
        self.log('loss',
                 sum(loss.values()),
                 batch_size=batch['images'].shape[0],
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.log('global_step', self.global_step, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        return sum(loss.values())

    def validation_step(self, batch, batch_idx):
        img_size = torch.tensor(tuple(batch['images'].shape[2:] * 2), device=batch['images'].device)

        outputs = self.net(batch['images'])
        pred_logits, pred_boxes = outputs["pred_logits"], outputs["pred_boxes"]
        pred_boxes = box_convert(pred_boxes, in_fmt='cxcywh', out_fmt='xyxy') * img_size

        pred_scores = pred_logits.sigmoid()
        pred_scores, index = torch.topk(pred_scores.flatten(1), self.hparams.config.num_top_queries, dim=-1)
        pred_labels = index - index // self.num_classes * self.num_classes
        index = index // self.num_classes
        pred_boxes = pred_boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, pred_boxes.shape[-1]))

        preds = [dict(labels=lab,
                      boxes=box,
                      scores=sco) for lab, box, sco in zip(pred_labels, pred_boxes, pred_scores)]
        targets = [dict(labels=target['labels'],
                        boxes=box_convert(target['boxes'], in_fmt='cxcywh', out_fmt='xyxy') * img_size) for target in batch['target']]

        self.map(preds, targets)

    def on_validation_epoch_end(self):
        metrics = self.map.compute()
        precision = metrics['precision'][0][50].mean()
        recall = metrics['recall'][0].mean()
        f1 = 2 * (precision * recall) / (precision + recall)
        self.log_dict({'mAP_50': metrics['map_50'],
                       'mAP_50_95': metrics['map'],
                       'precision': precision,
                       'recall': recall,
                       'f1': f1},
                      on_epoch=True,
                      prog_bar=True,
                      logger=True)
        self.map.reset()

    def setup(self, stage: Optional[str] = None):
        if stage in [None, 'fit']:
            if self.net is None:
                set_class_mapping = self.trainer.datamodule.train_set.dataset.class_mapping

                self.net = create_model('DFINEModel',
                                        model_variant=self.hparams.config.model_variant,
                                        image_size=self.trainer.datamodule.hparams.data_config.image_size,
                                        class_mapping=set_class_mapping)

                self.criterion = build_criterion(model_variant=self.hparams.config.model_variant,
                                                 class_mapping=set_class_mapping)

    def on_load_checkpoint(self, checkpoint):
        """
        Reconstruct the model from the spec here and not in setup() as
        otherwise the weight loading will fail.
        """
        if not isinstance(checkpoint['_module_config'], DFINESegmentationTrainingConfig):
            raise ValueError('Checkpoint is not a D-FINE model.')

        data_config = checkpoint['datamodule_hyper_parameters']['data_config']
        self.net = create_model('DFINEModel',
                                model_variant=checkpoint['_module_config'].model_variant,
                                image_size=self.trainer.datamodule.hparams.data_config.image_size,
                                class_mapping={'lines': data_config.line_class_mapping,
                                               'regions': data_config.region_class_mapping})

        self.criterion = build_criterion(model_variant=self.hparams.config.model_variant,
                                         class_mapping={'lines': data_config.line_class_mapping,
                                                        'regions': data_config.region_class_mapping})

    def on_save_checkpoint(self, checkpoint):
        """
        Save hyperparameters a second time so we can set parameters that
        shouldn't be overwritten in on_load_checkpoint.
        """
        checkpoint['_module_config'] = self.hparams.config

    @classmethod
    def load_from_weights(cls,
                          path: Union[str, 'PathLike'],
                          config: DFINESegmentationTrainingConfig) -> 'DFINESegmentationModel':
        """
        Initializes the module from a model weights file.
        """
        from kraken.models import load_models
        models = load_models(path, tasks=['segmentation'])
        if len(models) != 1:
            raise ValueError(f'Found {len(models)} segmentation models in model file.')
        return cls(config=config, model=models[0])

    def configure_callbacks(self):
        callbacks = []
        if self.hparams.config.quit == 'early':
            callbacks.append(EarlyStopping(monitor='mAP_50',
                                           mode='max',
                                           patience=self.hparams.config.lag,
                                           stopping_threshold=1.0))

        return callbacks

    # configuration of optimizers and learning rate schedulers
    # --------------------------------------------------------
    #
    # All schedulers are created internally with a frequency of step to enable
    # batch-wise learning rate warmup. In lr_scheduler_step() calls to the
    # scheduler are then only performed at the end of the epoch.
    def configure_optimizers(self):
        return configure_optimizer_and_lr_scheduler(self.hparams.config,
                                                    self.net.parameters(),
                                                    len_train_set=len(self.trainer.datamodule.train_set),
                                                    loss_tracking_mode='max')

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # linear warmup between 0 and the initial learning rate `lrate` in `warmup`
        # steps.
        if self.hparams.config.warmup and self.trainer.global_step < self.hparams.config.warmup:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.hparams)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.config.lrate

    def lr_scheduler_step(self, scheduler, metric):
        if not self.hparams.config.warmup or self.trainer.global_step >= self.hparams.config.warmup:
            # step OneCycleLR each batch if not in warmup phase
            if isinstance(scheduler, lr_scheduler.OneCycleLR):
                scheduler.step()
            # step every other scheduler epoch-wise
            elif self.trainer.is_last_batch:
                if metric is None:
                    scheduler.step()
                else:
                    scheduler.step(metric)
