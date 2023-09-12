from timm.models.efficientnet import tf_efficientnet_b7_ns, tf_efficientnet_b4_ns, tf_efficientnet_b3_ns
from torch import nn
from torch import optim
from functools import partial
from torch.nn.modules.pooling import AdaptiveAvgPool2d
from torch.nn.modules.linear import Linear
from typing import Any
from torch.optim.lr_scheduler import LRScheduler, PolynomialLR, ExponentialLR

EFFICIENTNET_CONFIGS = {
    "b4": {
        "init_func": tf_efficientnet_b4_ns,
        "init_args": {"pretrained": True, "drop_path_rate": 0.3},
        "num_features": 1792,
    },
    "b3": {
        "init_func": tf_efficientnet_b3_ns,
        "init_args": {"pretrained": True, "drop_path_rate": 0.2},
        "num_features": 1536,
    },
    "b7": {
        "init_func": tf_efficientnet_b7_ns,
        "init_args": {"pretrained": True, "drop_path_rate": 0.2},
        "num_features": 2560,
    },
}


class Classifier(nn.Module):
    def __init__(self, model_key: str) -> None:
        super().__init__()
        config = EFFICIENTNET_CONFIGS.get(model_key)
        if config is None:
            raise ValueError(f"No config found for model_key: {model_key}")

        self.model = config["init_func"](**config["init_args"])
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(config["num_features"], 1)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.fc(x)
        return x


def get_optimizer(
    model: nn.Module,
    optim_type: str,
    lr: float,
    momentum: float,
    weight_decay: float,
) -> optim.Optimizer:
    if optim_type == "SGD":
        optimizer = partial(optim.SGD, nesterov=True, momentum=momentum)
    elif optim_type == "Adam":
        optimizer = optim.Adam
    else:
        raise NotImplementedError("Must be one of Adam / SGD")

    return optimizer(params=model.parameters(), lr=lr, weight_decay=weight_decay)


def get_scheduler(optimizer: optim.Optimizer, scheduler_type: str, params: dict[str, Any]) -> LRScheduler:
    if scheduler_type == "poly":
        scheduler = PolynomialLR(optimizer=optimizer, **params)
    elif scheduler_type == "exp":
        scheduler = ExponentialLR(optimizer=optimizer, **params)
    else:
        raise NotImplementedError("Must be poly")

    return scheduler
