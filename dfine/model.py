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

from torch.optim import lr_scheduler

from typing import Literal

from torchvision.ops import box_convert
from torchmetrics.detection import MeanAveragePrecision

from dfine.configs import models
from dfine.modules import build_model, build_criterion

logger = logging.getLogger(__name__)


#@torch.compile()
def model_step(model, criterion, batch):
    o = model(batch['images'], targets=batch['target'])
    return criterion(outputs=o, targets=batch['target'])


class RegionDetectionModel(L.LightningModule):
    """
    A LightningModule encapsulating the training setup for a region object
    detection model.
    """
    def __init__(self,
                 num_classes: int,
                 model: Literal['nano', 'small', 'medium', 'large', 'extra_large'] = 'medium',
                 quit: Literal['fixed', 'early'] = 'fixed',
                 lag: int = 10,
                 optimizer: str = 'AdamW',
                 lr: float = 1e-4,
                 momentum: float = 0.9,
                 weight_decay: float = 1e-3,
                 schedule: Literal['cosine', 'exponential', 'step', 'reduceonplateau', 'constant'] = 'constant',
                 step_size: int = 10,
                 gamma: float = 0.1,
                 rop_factor: float = 0.1,
                 rop_patience: int = 5,
                 cos_t_max: float = 30,
                 cos_min_lr: float = 1e-4,
                 warmup: int = 100,
                 image_size: tuple[int, int] = (320, 320),
                 num_top_queries: int = 300,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        model_cfg = models[model]
        self.model = build_model(model_cfg=model_cfg,
                                 num_classes=num_classes,
                                 img_size=image_size)
        self.criterion = build_criterion(criterion_cfg=model_cfg,
                                         num_classes=num_classes)

        self.map = MeanAveragePrecision(box_format='xyxy',
                                        iou_type='bbox',
                                        extended_summary=True)

        self.model.train()
        self.criterion.train()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss = model_step(self.model, self.criterion, batch)
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
        img_size = torch.Tensor(tuple(batch['images'].shape[2:] * 2), device=batch['images'].device)

        outputs = self.model(batch['images'])
        pred_logits, pred_boxes = outputs["pred_logits"], outputs["pred_boxes"]
        pred_boxes = box_convert(pred_boxes, in_fmt='cxcywh', out_fmt='xyxy') * img_size

        pred_scores = torch.sigmoid(pred_logits)
        pred_scores, index = torch.topk(pred_scores.flatten(1), self.hparams.num_top_queries, dim=-1)
        pred_labels = index - index // self.hparams.num_classes * self.hparams.num_classes
        index = index // self.hparams.num_classes
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

    def save_checkpoint(self, filename):
        self.trainer.save_checkpoint(filename)

    # configuration of optimizers and learning rate schedulers
    # --------------------------------------------------------
    #
    # All schedulers are created internally with a frequency of step to enable
    # batch-wise learning rate warmup. In lr_scheduler_step() calls to the
    # scheduler are then only performed at the end of the epoch.
    def configure_optimizers(self):
        return _configure_optimizer_and_lr_scheduler(self.hparams,
                                                     self.model,
                                                     loss_tracking_mode='min')

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # linear warmup between 0 and the initial learning rate `lr` in `warmup`
        # steps.
        if self.hparams.warmup and self.trainer.global_step < self.hparams.warmup:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.hparams.warmup)
            for pg in optimizer.param_groups:
                if 'lr_scale' in pg:
                    lr_scale = pg['lr_scale'] * lr_scale
                if self.hparams.optimizer not in ['Adam8bit', 'Adam4bit', 'AdamW8bit', 'AdamW4bit', 'AdamWFp8']:
                    pg['lr'] = lr_scale * self.hparams.lr
                else:
                    pg['lr'].fill_(lr_scale * self.hparams.lr)

    def lr_scheduler_step(self, scheduler, metric):
        if not self.hparams.warmup or self.trainer.global_step >= self.hparams.warmup:
            # step OneCycleLR each batch if not in warmup phase
            if isinstance(scheduler, lr_scheduler.OneCycleLR):
                scheduler.step()
            # step every other scheduler epoch-wise
            elif self.trainer.is_last_batch:
                if metric is None:
                    scheduler.step()
                else:
                    scheduler.step(metric)


def _configure_optimizer_and_lr_scheduler(hparams, model, loss_tracking_mode='min'):
    optimizer = hparams.get("optimizer")
    lr = hparams.get("lr")
    momentum = hparams.get("momentum")
    weight_decay = hparams.get("weight_decay")
    schedule = hparams.get("schedule")
    gamma = hparams.get("gamma")
    cos_t_max = hparams.get("cos_t_max")
    cos_min_lr = hparams.get("cos_min_lr")
    step_size = hparams.get("step_size")
    rop_factor = hparams.get("rop_factor")
    rop_patience = hparams.get("rop_patience")
    completed_epochs = hparams.get("completed_epochs")

    param_groups = filter(lambda p: p.requires_grad, model.parameters())

    # XXX: Warmup is not configured here because it needs to be manually done in optimizer_step()
    logger.debug(f'Constructing {optimizer} optimizer (lr: {lr}, momentum: {momentum})')
    if optimizer in ['Adam', 'AdamW']:
        optim = getattr(torch.optim, optimizer)(param_groups, lr=lr, weight_decay=weight_decay)
    elif optimizer in ['Adam8bit', 'Adam4bit', 'AdamW8bit', 'AdamW4bit', 'AdamWFp8']:
        import torchao.prototype.low_bit_optim
        optim = getattr(torchao.prototype.low_bit_optim, optimizer)(param_groups, lr=lr, weight_decay=weight_decay)
    elif optimizer == 'Mars':
        from timm.optim import Mars
        optim = Mars(param_groups, lr=lr, weight_decay=weight_decay, caution=True)
    else:
        optim = getattr(torch.optim, optimizer)(param_groups,
                                                lr=lr,
                                                momentum=momentum,
                                                weight_decay=weight_decay)
    lr_sched = {}
    if schedule == 'exponential':
        lr_sched = {'scheduler': lr_scheduler.ExponentialLR(optim, gamma, last_epoch=completed_epochs-1),
                    'interval': 'step'}
    elif schedule == 'cosine':
        lr_sched = {'scheduler': lr_scheduler.CosineAnnealingLR(optim,
                                                                cos_t_max,
                                                                cos_min_lr,
                                                                last_epoch=completed_epochs-1),
                    'interval': 'step'}
    elif schedule == 'step':
        lr_sched = {'scheduler': lr_scheduler.StepLR(optim, step_size, gamma, last_epoch=completed_epochs-1),
                    'interval': 'step'}
    elif schedule == 'reduceonplateau':
        lr_sched = {'scheduler': lr_scheduler.ReduceLROnPlateau(optim,
                                                                mode=loss_tracking_mode,
                                                                factor=rop_factor,
                                                                patience=rop_patience),
                    'interval': 'step'}
    elif schedule != 'constant':
        raise ValueError(f'Unsupported learning rate scheduler {schedule}.')

    ret = {'optimizer': optim}
    if lr_sched:
        ret['lr_scheduler'] = lr_sched

    if schedule == 'reduceonplateau':
        lr_sched['monitor'] = 'val_accuracy'
        lr_sched['strict'] = False
        lr_sched['reduce_on_plateau'] = True

    return ret
