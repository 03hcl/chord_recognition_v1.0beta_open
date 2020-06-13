from typing import Dict

import torch

from .output_key_enum import OutputKeyEnum


class OutputTensor:

    def __init__(self, raw: torch.Tensor):
        self.raw: torch.Tensor = raw
        self.dict: Dict[OutputKeyEnum, torch.Tensor] \
            = dict((i, raw[:, int(i)]) for i in OutputKeyEnum.__members__.values())
