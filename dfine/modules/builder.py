from .matcher import HungarianMatcher

from .dfine import DFINE
from .hgnetv2 import HGNetv2
from .decoder import DFINETransformer
from .criterion import DFINECriterion
from .hybrid_encoder import HybridEncoder

from dfine.utils import load_tuning_state

from os import PathLike
from typing import Optional

def build_model(model_cfg: dict,
                num_classes: int,
                img_size: Optional[tuple[int, int]] = None,
                model_weights: Optional[PathLike] = None) -> DFINE:
    model_cfg["HybridEncoder"]["eval_spatial_size"] = img_size
    model_cfg["DFINETransformer"]["eval_spatial_size"] = img_size

    backbone = HGNetv2(**model_cfg["HGNetv2"])
    encoder = HybridEncoder(**model_cfg["HybridEncoder"])
    decoder = DFINETransformer(num_classes=num_classes, **model_cfg["DFINETransformer"])

    model = DFINE(backbone, encoder, decoder)

    if model_weights:
        if not Path(model_weights).exists():
            raise FileNotFoundError(f"{model_weights} does not exist")
        model = load_tuning_state(model, model_weights)
    return model


def build_criterion(criterion_cfg: dict,
                    num_classes: int) -> DFINECriterion:
    matcher = HungarianMatcher(**criterion_cfg["matcher"])
    return DFINECriterion(matcher,
                          num_classes=num_classes,
                          **criterion_cfg["DFINECriterion"])


def build_optimizer(model: DFINE,
                    lr,
                    backbone_lr,
                    betas,
                    weight_decay,
                    base_lr):
    backbone_exclude_norm = []
    backbone_norm = []
    encdec_norm_bias = []
    rest = []

    for name, param in model.named_parameters():
        # Group 1 and 2: "backbone" in name
        if "backbone" in name:
            if "norm" in name or "bn" in name:
                # Group 2: backbone + norm/bn
                backbone_norm.append(param)
            else:
                # Group 1: backbone but not norm/bn
                backbone_exclude_norm.append(param)

        # Group 3: "encoder" or "decoder" plus "norm"/"bn"/"bias"
        elif ("encoder" in name or "decoder" in name) and (
            "norm" in name or "bn" in name or "bias" in name
        ):
            encdec_norm_bias.append(param)

        else:
            rest.append(param)

    group1 = {"params": backbone_exclude_norm, "lr": backbone_lr, "initial_lr": backbone_lr}
    group2 = {
        "params": backbone_norm,
        "lr": backbone_lr,
        "weight_decay": 0.0,
        "initial_lr": backbone_lr,
    }
    group3 = {"params": encdec_norm_bias, "weight_decay": 0.0, "lr": base_lr, "initial_lr": base_lr}
    group4 = {"params": rest, "lr": base_lr, "initial_lr": base_lr}

    param_groups = [group1, group2, group3, group4]

    return optim.AdamW(param_groups, lr=lr, betas=betas, weight_decay=weight_decay)
