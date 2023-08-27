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
    optim_config: OptimizerConfig
    scheduler_config: SchedulerConfig

default_config = TrainingConfig(
    model_key="b4",
    optim_config=OptimizerConfig(optim_type="SGD", learning_rate=0.01, momentum=0.9, weight_decay=1e-4),
    scheduler_config=SchedulerConfig(scheduler_type="poly", params={"power": 0.9, "total_iters": 30})
)


def get_config(config_name: str) -> TrainingConfig:
    configs = {
        "default_config": default_config
    }

    config = configs[config_name]

    return config