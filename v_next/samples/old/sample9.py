# import os

import torch

from v_next.config import ConfigTest

# noinspection PyUnresolvedReferences
from ...src.utils import Device, iterate_file_path_set, UtilLogger
# noinspection PyUnresolvedReferences
from ...src.cnn import Config, create_dataset


def main():

    # lab_dirs = [
    #     "*" + os.extsep + "lab",
    #     "**" + os.sep + "chordlab" + os.sep + "**" + os.sep + "*" + os.extsep + "lab",
    # ]
    #
    # for k, v in iterate_file_path_set(
    #         "." + os.sep + "data" + os.sep + "raw",
    #         ["**" + os.sep + "*" + os.extsep + "wav", ], {"lab": lab_dirs},
    #         key_name="wav", raises_error=False):
    #     print(k)
    #     print(v)

    present_config: Config = ConfigTest(__file__)
    # for file in present_config.train_file_path_pairs:
    #     print(file)

    # noinspection PyUnusedLocal
    logger: UtilLogger = UtilLogger(present_config)
    # noinspection PyUnusedLocal
    device: Device = Device(gpu_number=0)
    # noinspection PyUnusedLocal
    target: torch.Tensor

    # tensors = create_dataset(present_config, present_config.train_file_path_pairs, logger=logger)
    # for t in tensors:
    #     print(t.shape)
    #     # print(t)
    #     # print()


if __name__ == '__main__':
    main()
