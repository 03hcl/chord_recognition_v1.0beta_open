import os

from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from v_next.src.utils import ModuleConfig, TrainConfig
# noinspection PyUnresolvedReferences
from v_next.src.utils.torchhelper import OptunaParameter, OptunaSuggestion

from v_next.src.cnn import CNN, Config, TestResourceConfig, TrainResourceConfig


class ConfigTest20200601(Config):
    def __init__(self, root_dir: str, *args, **kwargs):
        super(ConfigTest20200601, self).__init__(
            model=ModuleConfig(CNN, input_time_width=9),
            criterion=ModuleConfig(CrossEntropyLoss),
            optimizer=ModuleConfig(Adam, lr=2e-3, eps=2e-6),
            train=TrainConfig(batch_size=1000, number_of_epochs=50, progress_epoch=1, temporary_save_epoch=5),
            train_resource=TrainResourceConfig(chord_directories=("*" + os.extsep + "lab", )),
            test_resource=TestResourceConfig(enforced_recreate=False),
            stride=0.05, f_min=440 / 8, f_max=440 * 16, b=24,
            root_dir=root_dir,
            *args, **kwargs)
