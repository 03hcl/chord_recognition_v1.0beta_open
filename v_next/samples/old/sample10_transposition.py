# import os

import torch

from v_next.config import ConfigTest

# noinspection PyUnresolvedReferences
from ...src.utils import Device, iterate_file_path_set, UtilLogger
# noinspection PyUnresolvedReferences
from ...src.cnn import Config, create_dataset


def main():

    present_config: Config = ConfigTest(__file__)
    # noinspection PyUnusedLocal
    logger: UtilLogger = UtilLogger(present_config)
    # noinspection PyUnusedLocal
    device: Device = Device(gpu_number=0)
    # noinspection PyUnusedLocal
    target: torch.Tensor

    # tensors = create_dataset(present_config, present_config.train_file_path_pairs, logger=logger)
    # for t in tensors:
    #     print(t.shape)


if __name__ == '__main__':
    main()
