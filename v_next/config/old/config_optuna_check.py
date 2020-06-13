import os

from optuna.pruners import PercentilePruner

from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from v_next.src.utils import ModuleConfig, TrainConfig
# noinspection PyUnresolvedReferences
from v_next.src.utils.torchhelper import OptunaParameter, OptunaSuggestion

from v_next.src.cnn import CNN, Config, TestResourceConfig, TrainResourceConfig


class ConfigOptunaCheck(Config):
    def __init__(self, root_dir: str, *args, **kwargs):
        super(ConfigOptunaCheck, self).__init__(
            model=ModuleConfig(CNN, input_time_width=9, input_freq_height=217),
            criterion=ModuleConfig(CrossEntropyLoss),
            optimizer=ModuleConfig(
                Adam,
                lr=OptunaParameter(OptunaSuggestion.LogUniform, name="lr", low=1e-4, high=1e-1),
                eps=OptunaParameter(OptunaSuggestion.LogUniform, name="eps", low=1e-8, high=1e-4)),
            train=TrainConfig(batch_size=1000, number_of_epochs=3, progress_epoch=1, temporary_save_epoch=1,
                              optuna_number_of_trials=3, optuna_pre_trials=2,
                              optuna_pruner=PercentilePruner(200 / 3, n_startup_trials=5, n_warmup_steps=30)),
            # optimizer=ModuleConfig(Adam, lr=1e-3, eps=5e-5),
            # train=TrainConfig(batch_size=1000, number_of_epochs=3, progress_epoch=1, temporary_save_epoch=1),
            train_resource=TrainResourceConfig(chord_directories=("*" + os.extsep + "lab", )),
            test_resource=TestResourceConfig(enforced_recreate=False),
            stride=0.05,
            root_dir=root_dir,
            *args, **kwargs)
