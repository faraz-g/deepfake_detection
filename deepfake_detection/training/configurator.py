from pydantic import BaseModel
from typing import Any

class OptimizerConfig(BaseModel):
    optim_type: str
    learning_rate: float
    momentum: float
    weight_decay: float

class SchedulerConfig(BaseModel):
    scheduler_type: str
    params: dict[str, Any]

class TrainingConfig(BaseModel):
    model_key: str
    seed: int
    batch_size: int
    batches_per_epoch: int 
    max_epochs: int
    evaluation_frequency: int
    img_height: int 
    img_width: int
    optim_config: OptimizerConfig
    scheduler_config: SchedulerConfig

default_config = TrainingConfig(
    model_key="b4",
    seed=111,
    batch_size=20,
    batches_per_epoch=5000,
    max_epochs=80,
    evaluation_frequency=1,
    img_height=380,
    img_width=380,
    optim_config=OptimizerConfig(optim_type="SGD", learning_rate=0.01, momentum=0.9, weight_decay=1e-4),
    scheduler_config=SchedulerConfig(scheduler_type="poly", params={"total_iters" : 100000, "power" : 0.9})
)


def get_config(config_name: str) -> TrainingConfig:
    configs = {
        "default_config": default_config
    }

    config = configs[config_name]

    return config