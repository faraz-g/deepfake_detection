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
    early_stopping_threshold: int
    optim_config: OptimizerConfig
    scheduler_config: SchedulerConfig


baseline_config = TrainingConfig(
    model_key="b4",
    seed=111,
    batch_size=20,
    batches_per_epoch=5000,
    max_epochs=80,
    evaluation_frequency=1,
    img_height=380,
    img_width=380,
    early_stopping_threshold=3,
    optim_config=OptimizerConfig(optim_type="Adam", learning_rate=0.001, momentum=0.9, weight_decay=1e-4),
    scheduler_config=SchedulerConfig(scheduler_type="cosine", params={"T_max": 30, "eta_min": 0.0005}),
)

default_config = TrainingConfig(
    model_key="b4",
    seed=111,
    batch_size=20,
    batches_per_epoch=5000,
    max_epochs=80,
    evaluation_frequency=1,
    img_height=380,
    img_width=380,
    early_stopping_threshold=3,
    optim_config=OptimizerConfig(optim_type="Adam", learning_rate=0.005, momentum=0.9, weight_decay=1e-4),
    scheduler_config=SchedulerConfig(scheduler_type="cosine", params={"T_max": 60, "eta_min": 0}),
)

default_config_b7 = TrainingConfig(
    model_key="b7",
    seed=111,
    batch_size=8,
    batches_per_epoch=5000,
    max_epochs=80,
    evaluation_frequency=1,
    img_height=380,
    img_width=380,
    early_stopping_threshold=3,
    optim_config=OptimizerConfig(optim_type="Adam", learning_rate=0.001, momentum=0.9, weight_decay=1e-4),
    scheduler_config=SchedulerConfig(scheduler_type="cosine", params={"T_max": 30, "eta_min": 0.0005}),
)


def get_config(config_name: str) -> TrainingConfig:
    configs = {
        "default_config": default_config,
        "default_config_b7": default_config_b7,
        "baseline_config": baseline_config,
    }

    config = configs[config_name]

    return config
