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
dfine.cli.train
~~~~~~~~~~~~~~~

Command line driver for segmentation training
"""
import logging

import click

from threadpoolctl import threadpool_limits

from .util import _expand_gt, _validate_manifests, message, to_ptl_device, _validate_merging

logging.captureWarnings(True)
logger = logging.getLogger('dfine')

# suppress worker seeding message
logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.ERROR)


def _cls_lst_to_dict(ctx, param, values):
    cls_dict = {val: idx for idx, val in enumerate(values)}
    return cls_dict if cls_dict else None


@click.command('train')
@click.option('--load', default=None, type=click.Path(exists=True), help='Path to checkpoint/safetensors to load')
@click.option('--resume', default=None, type=click.Path(exists=True), help='Path to checkpoint to resume from')
@click.option('-i', '--image-size', type=tuple[int, int], help='Input image dimensions as (height, width) in pixels.')
@click.option('-B', '--batch-size', type=int, help='batch sample size')
@click.option('-o', '--output', type=click.Path(), help='Output model file')
@click.option('-F', '--freq', type=float,
        help='Model saving and report generation frequency in epochs '
             'during training. If frequency is >1 it must be an integer, '
             'i.e. running validation every n-th epoch.')
@click.option('-N',
              '--epochs',
              type=int,
              help='Number of epochs to train for')
@click.option('--freeze-encoder/--no-freeze-encoder', help='Switch to freeze the encoder')
@click.option('--optimizer',
              type=click.Choice(['Adam',
                                 'AdamW',
                                 'SGD',
                                 'Mars',
                                 'Adam8bit',
                                 'AdamW8bit',
                                 'Adam4bit',
                                 'AdamW4bit']),
              help='Select optimizer')
@click.option('-r', '--lrate', type=float, help='Learning rate')
@click.option('-m', '--momentum', type=float, help='Momentum')
@click.option('-w', '--weight-decay', type=float, help='Weight decay')
@click.option('--gradient-clip-val', type=float, help='Gradient clip value')
@click.option('--warmup', type=int, help='Number of steps to ramp up to `lrate` initial learning rate.')
@click.option('--schedule',
              type=click.Choice(['constant',
                                 '1cycle',
                                 'exponential',
                                 'cosine',
                                 'step',
                                 'reduceonplateau']),
              help='Set learning rate scheduler. For 1cycle, cycle length is determined by the `--epoch` option.')
@click.option('-g',
              '--gamma',
              type=float,
              help='Decay factor for exponential, step, and reduceonplateau learning rate schedules')
@click.option('-ss',
              '--step-size',
              type=click.IntRange(1),
              help='Number of validation runs between learning rate decay for exponential and step LR schedules')
@click.option('--sched-patience',
              type=click.IntRange(1),
              help='Minimal number of validation runs between LR reduction for reduceonplateau LR schedule.')
@click.option('--cos-max',
              type=click.IntRange(1),
              help='Epoch of minimal learning rate for cosine LR scheduler.')
@click.option('--cos-min-lr',
              type=float,
              help='Minimal final learning rate for cosine LR scheduler.')
@click.option('-t', '--training-files', multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data')
@click.option('-e', '--evaluation-files', default=None, multiple=True,
        callback=_validate_manifests, type=click.File(mode='r', lazy=True),
        help='File(s) with paths to evaluation data.')
@click.option('--class-mapping', multiple=True, help='List of classes.', callback=_cls_lst_to_dict)
@click.option('-vr', '--valid-regions', multiple=True,
        help='Valid region types in training data. May be used multiple times.')
@click.option('-mr',
        '--merge-regions',
        help='Region merge mapping. One or more mappings of the form `$target:$src` where $src is merged into $target.',
        multiple=True,
        callback=_validate_merging)
@click.option('--merge-all-regions', help='Merges all region types into the argument identifiers')
@click.option('--accumulate-grad-batches', type=click.IntRange(1), help='Number of batches to accumulate gradient across.')
@click.option('--validate-before-train/--no-validate-before-train', default=True, help='Enables validation run before first training run.')
@click.argument('ground_truth', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
@click.pass_context
def train(ctx, **kwargs):
    """
    Trains an object detection model from XML facsimile files.
    """
    params = ctx.params

    if not (0 <= params['freq'] <= 1) and params['freq'] % 1.0 != 0:
        raise click.BadOptionUsage('freq', 'freq needs to be either in the interval [0,1.0] or a positive integer.')

    if sum(map(bool, [params['load'], params['resume']])) > 1:
        raise click.BadOptionsUsage('load', 'load/resume options are mutually exclusive.')

    import torch

    from dfine.dataset import RegionDetectionDataModule
    from dfine.model import RegionDetectionModel

    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import RichModelSummary, ModelCheckpoint, RichProgressBar

    torch.set_float32_matmul_precision('high')

    ground_truth = list(params['ground_truth'])

    # merge training_files into ground_truth list
    ground_truth.extend(params.get('training_files', []))
    if len(ground_truth) == 0:
        raise click.UsageError('No training data was provided to the train command. Use `-t` or the `ground_truth` argument.')

    params.setdefault('valid_regions', None)

    try:
        accelerator, device = to_ptl_device(ctx.meta['device'])
    except Exception as e:
        raise click.BadOptionUsage('device', str(e))

    if params['freq'] > 1:
        val_check_interval = {'check_val_every_n_epoch': int(params['freq'])}
    else:
        val_check_interval = {'val_check_interval': params['freq']}

    if not params['valid_regions']:
        params['valid_regions'] = None

    if params['resume']:
        data_module = RegionDetectionDataModule.load_from_checkpoint(resume)
    else:
        data_module = RegionDetectionDataModule(training_data=ground_truth,
                                                evaluation_data=params.pop('evaluation_files'),
                                                num_workers=ctx.meta['workers'],
                                                **params)

    cbs = [RichModelSummary(max_depth=2)]

    checkpoint_callback = ModelCheckpoint(dirpath=params['output'],
                                          save_top_k=10,
                                          monitor='global_step',
                                          mode='max',
                                          auto_insert_metric_name=False,
                                          filename='checkpoint_{epoch:02d}-{val_metric:.4f}')

    cbs.append(checkpoint_callback)
    if not ctx.meta['verbose']:
        cbs.append(RichProgressBar(leave=True))

    trainer = Trainer(accelerator=accelerator,
                      devices=device,
                      precision=ctx.meta['precision'],
                      max_epochs=params['epochs'],
                      enable_progress_bar=True if not ctx.meta['verbose'] else False,
                      deterministic=ctx.meta['deterministic'],
                      enable_model_summary=False,
                      accumulate_grad_batches=params['accumulate_grad_batches'],
                      callbacks=cbs,
                      gradient_clip_val=params['gradient_clip_val'],
                      num_sanity_val_steps=0,
                      **val_check_interval)

    with trainer.init_module(empty_init=True if (params['load'] or params['resume']) else False):
        if params['load']:
            message(f'Loading from checkpoint {params["load"]}.')
            model = RegionDetectionModel.load(**params)
        elif params['resume']:
            message(f'Resuming from checkpoint {params["resume"]}.')
            model = RegionDetectionModel.load_from_checkpoint(params['resume'])
        else:
            model = RegionDetectionModel(num_classes=data_module.num_classes, **params)

    with threadpool_limits(limits=ctx.meta['threads']):
        if params['resume']:
            trainer.fit(model, data_module, ckpt_path=params['resume'])
        else:
            if params['validate_before_train']:
                trainer.validate(model, data_module)
            trainer.fit(model, data_module)

    if not model.current_epoch:
        logger.warning('Training aborted before end of first epoch.')
        ctx.exit(1)
