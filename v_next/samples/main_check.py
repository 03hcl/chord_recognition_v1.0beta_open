# noinspection PyUnresolvedReferences
from ..config import ConfigCheck, ConfigDefaultCheck, ConfigOptunaCheck, ConfigTransformerCheck

# noinspection PyUnresolvedReferences
# from ..src.utils import Device, iterate_file_path_set, UtilLogger
# from ..src.cnn import Config, Trainer, Predictor
from ..src.transformer import Config, Trainer, Predictor


def main():

    # present_config: Config = ConfigCheck(__file__)
    # present_config: Config = ConfigDefaultCheck(__file__)
    # present_config: Config = ConfigOptunaCheck(__file__)
    present_config: Config = ConfigTransformerCheck(__file__)

    device_kwargs = {"gpu_number": 0}
    Trainer.run(config=present_config, device_kwargs=device_kwargs,
                visualizes_loss_on_logscale=True)
    Predictor.run(config=present_config, device_kwargs=device_kwargs,
                  use_validation_data=True)


if __name__ == '__main__':
    # import matplotlib
    # matplotlib.use("Agg")
    main()
