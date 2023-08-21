from timm.models.efficientnet import tf_efficientnet_b4_ns, tf_efficientnet_b3_ns
from torch import nn
from torch import optim
from functools import partial
from training.tools.schedulers import ExponentialLRScheduler, PolyLR, LRStepScheduler

MODEL_CONFIGS = {
    "b4" : {
        "init_func" : tf_efficientnet_b4_ns,
        "init_args" : {"pretrained": True}
    },
    "b3" : {
        "init_func" : tf_efficientnet_b3_ns,
        "init_args" : {"pretrained": True}
    }
}

class Classifier(nn.Module):
    def __init__(self, model_key: str) -> None:
        super().__init__()
        config = MODEL_CONFIGS.get(model_key)
        self.model = config['init_func'](**config['init_args'])

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
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
    
    return optimizer(
        params=model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )