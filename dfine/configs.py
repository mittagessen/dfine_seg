from kraken.configs import TrainingConfig, SegmentationTrainingDataConfig


class DFINESegmentationTrainingConfig(TrainingConfig):
    """
    Base configuration for training a D-FINE segmentation model.

    Args:
        model_variant (Literal[], defaults to medium):
            D-FINE model variant
        num_top_queries (int, defaults to 300):
    """
    def __init__(self, **kwargs):
        self.model_variant = kwargs.pop('model_variant', 'medium')
        self.num_top_queries = kwargs.pop('num_top_queries', 300)

        kwargs.setdefault('quit', 'fixed')
        kwargs.setdefault('epochs', 50)
        kwargs.setdefault('lrate', 1e-4)
        kwargs.setdefault('weight_decay', 1e-5)
        super().__init__(**kwargs)


class DFINESegmentationTrainingDataConfig(SegmentationTrainingDataConfig):
    """
    Base data configuration for a D-FINE segmentation model.
    """
    def __init__(self, **kwargs):
        self.image_size = kwargs.pop('image_size', (1280, 1280))
        kwargs.setdefault('batch_size', 16)

        super().__init__(**kwargs)


base_cfg = {
    "HGNetv2": {
        "pretrained": False,
        "local_model_dir": "weight/hgnetv2/",
        "freeze_stem_only": True,
    },
    "HybridEncoder": {
        "num_encoder_layers": 1,
        "nhead": 8,
        "dropout": 0.0,
        "enc_act": "gelu",
        "act": "silu",
    },
    "DFINETransformer": {
        "eval_idx": -1,
        "num_queries": 300,
        "num_denoising": 100,
        "label_noise_ratio": 0.5,
        "box_noise_scale": 1.0,
        "reg_max": 32,
        "layer_scale": 1,
        "cross_attn_method": "default",
        "query_select_method": "default",
    },
    "DFINECriterion": {
        "weight_dict": {
            "loss_vfl": 1,
            "loss_bbox": 5,
            "loss_giou": 2,
            "loss_fgl": 0.15,
            "loss_ddf": 1.5,
        },
        "losses": ["vfl", "boxes", "local"],
        "alpha": 0.75,
        "gamma": 2.0,
        "reg_max": 32,
    },
    "matcher": {
        "weight_dict": {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
        "alpha": 0.25,
        "gamma": 2.0,
        "use_focal_loss": True,
    },
}

sizes_cfg = {
    "nano": {
        "HGNetv2": {
            "name": "B0",
            "return_idx": [2, 3],
            "freeze_at": -1,
            "freeze_norm": False,
            "use_lab": True,
        },
        "HybridEncoder": {
            "in_channels": [512, 1024],
            "feat_strides": [16, 32],
            "hidden_dim": 128,
            "use_encoder_idx": [1],
            "dim_feedforward": 512,
            "expansion": 0.34,
            "depth_mult": 0.5,
        },
        "DFINETransformer": {
            "feat_channels": [128, 128],
            "feat_strides": [16, 32],
            "hidden_dim": 128,
            "num_levels": 2,
            "num_layers": 3,
            "reg_scale": 4,
            "num_points": [6, 6],
            "dim_feedforward": 512,
        },
    },
    "small": {
        "HGNetv2": {
            "name": "B0",
            "return_idx": [1, 2, 3],
            "freeze_at": -1,
            "freeze_norm": False,
            "use_lab": True,
        },
        "HybridEncoder": {
            "in_channels": [256, 512, 1024],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 256,
            "use_encoder_idx": [2],
            "dim_feedforward": 1024,
            "expansion": 0.5,
            "depth_mult": 0.34,
        },
        "DFINETransformer": {
            "feat_channels": [256, 256, 256],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 256,
            "num_levels": 3,
            "num_layers": 3,
            "reg_scale": 4,
            "num_points": [3, 6, 3],
        },
    },
    "medium": {
        "HGNetv2": {
            "name": "B2",
            "return_idx": [1, 2, 3],
            "freeze_at": -1,
            "freeze_norm": False,
            "use_lab": True,
        },
        "HybridEncoder": {
            "in_channels": [384, 768, 1536],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 256,
            "use_encoder_idx": [2],
            "dim_feedforward": 1024,
            "expansion": 1.0,
            "depth_mult": 0.67,
        },
        "DFINETransformer": {
            "feat_channels": [256, 256, 256],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 256,
            "dim_feedforward": 1024,
            "num_levels": 3,
            "num_layers": 4,
            "reg_scale": 4,
            "num_points": [3, 6, 3],
        },
    },
    "large": {
        "HGNetv2": {
            "name": "B4",
            "return_idx": [1, 2, 3],
            "freeze_at": 0,
            "freeze_norm": True,
            "use_lab": False,
        },
        "HybridEncoder": {
            "in_channels": [512, 1024, 2048],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 256,
            "use_encoder_idx": [2],
            "dim_feedforward": 1024,
            "expansion": 1.0,
            "depth_mult": 1.0,
        },
        "DFINETransformer": {
            "feat_channels": [256, 256, 256],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 256,
            "dim_feedforward": 1024,
            "num_levels": 3,
            "num_layers": 6,
            "reg_scale": 4,
            "num_points": [3, 6, 3],
        },
    },
    "extra_large": {
        "HGNetv2": {
            "name": "B5",
            "return_idx": [1, 2, 3],
            "freeze_at": 0,
            "freeze_norm": True,
            "use_lab": False,
        },
        "HybridEncoder": {
            "in_channels": [512, 1024, 2048],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 384,
            "use_encoder_idx": [2],
            "dim_feedforward": 2048,
            "expansion": 1.0,
            "depth_mult": 1.0,
        },
        "DFINETransformer": {
            "feat_channels": [384, 384, 384],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 256,
            "dim_feedforward": 1024,
            "num_levels": 3,
            "num_layers": 6,
            "reg_scale": 8,
            "num_points": [3, 6, 3],
        },
    },
}

AUGMENTATION_CONFIG = {'rotation_degree': 10,
                       'rotation_p': 0.0,
                       'multiscale_prob': 0.0,
                       'rotate_90': 0.05,
                       'left_right_flip': 0.3,
                       'up_down_flip': 0.0,
                       'to_gray': 0.01,
                       'blur': 0.01,
                       'gamma': 0.02,
                       'brightness': 0.02,
                       'noise': 0.01,
                       'coarse_dropout': 0.0,
                       'mosaic_prob': 0.8,
                       'no_mosaic_epochs': 5,
                       'mosaic_scale': [0.5, 1.5],
                       'mosaic_degrees': 0.0,
                       'mosaic_translate': 0.2,
                       'mosaic_shear': 2.0
                      }

def merge_configs(base, size_specific):
    result = {**base}
    for key, value in size_specific.items():
        if key in result and isinstance(result[key], dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


models = {size: merge_configs(base_cfg, config) for size, config in sizes_cfg.items()}
