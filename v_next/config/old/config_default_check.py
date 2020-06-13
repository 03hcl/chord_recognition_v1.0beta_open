from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from v_next.src.utils import ModuleConfig, TrainConfig
# noinspection PyUnresolvedReferences
from v_next.src.utils.torchhelper import OptunaParameter, OptunaSuggestion

from v_next.src.cnn import CNN, Config, TestResourceConfig, TrainResourceConfig


class ConfigDefaultCheck(Config):
    def __init__(self, root_dir: str, *args, **kwargs):
        super(ConfigDefaultCheck, self).__init__(
            model=ModuleConfig(CNN, input_time_width=9, input_freq_height=217),
            criterion=ModuleConfig(CrossEntropyLoss),
            optimizer=ModuleConfig(Adam, lr=1e-3, eps=5e-5),
            train=TrainConfig(batch_size=1000, number_of_epochs=3, progress_epoch=1, temporary_save_epoch=1),
            train_resource=TrainResourceConfig(sound_gaps={
                "love_me_do": -0.7,
                "can't_buy_me_love": -0.4,
                "a_hard_day's_night": 0.3,
                "eight_days_a_week": -0.75,
                "ticket_to_ride": -0.2,
                "help!": 0.75,
                "yesterday": -0.1,
                "yellow_submarine": -0.45,
                "eleanor_rigby": -0.4,
                "penny_lane": 0.25,
                "all_you_need_is_love": 0.2,
                "hello_goodbye": 0.3,
                "get_back": 20,
                "something": 0,
                "come_together": 0.6,
                "let_it_be": -0.5,
                "the_long_and_winding_road": -0.1,
            }),
            test_resource=TestResourceConfig(enforced_recreate=False),
            stride=0.05,
            root_dir=root_dir,
            *args, **kwargs)
