# noinspection PyUnresolvedReferences
from ..config import ConfigTest, ConfigTest20200601, \
    ConfigTransformerTest, ConfigTransformerTestMixed, ConfigTransformerOptunaTest

# from ..src.utils import Device, iterate_file_path_set, UtilLogger
# noinspection PyUnresolvedReferences
# from ..src.cnn import Config, Trainer, Predictor
from ..src.transformer import Config, Trainer, Predictor


def main():

    # present_config: Config = ConfigTest(__file__)
    # present_config: Config = ConfigTest20200601(__file__)
    # present_config: Config = ConfigTransformerTest(__file__)
    present_config: Config = ConfigTransformerTestMixed(__file__)
    # present_config: Config = ConfigTransformerOptunaTest(__file__)

    device_kwargs = {"gpu_number": 0}
    # Trainer.run(config=present_config, device_kwargs=device_kwargs,
    #             visualizes_loss_on_logscale=True)
    Predictor.run(config=present_config, device_kwargs=device_kwargs,
                  use_validation_data=True)


if __name__ == '__main__':
    main()
