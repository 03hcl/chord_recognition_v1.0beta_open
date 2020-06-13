from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn.functional import max_pool1d, relu    # , max_pool2d

from ..data_model import OutputKeyEnum  # , OutputTensor
from ..utils import UtilLogger
from ..utils.music import consts

from .exceptions import NotOddTimeWidthError


class CNN(nn.Module):

    def __init__(self, input_time_width: int, input_freq_height: int,
                 *, logger: Optional[UtilLogger] = None):

        super(CNN, self).__init__()
        if input_time_width % 2 == 0:
            raise NotOddTimeWidthError

        self.input_time_width: int = input_time_width
        self.input_freq_height: int = input_freq_height

        conv2ds: List[nn.Conv2d] = []
        batchnorm2ds: List[nn.BatchNorm2d] = []
        channels: Tuple[int, int] = (1, 32)

        def proceed_channels_2d(present: Tuple[int, int]) -> Tuple[int, int]:
            return present[1], present[1] + 16

        while input_time_width > 1:
            conv2ds.append(nn.Conv2d(*channels, kernel_size=(3, 3), padding=(0, 1)))
            channels = proceed_channels_2d(channels)
            input_time_width -= 2
            batchnorm2ds.append(nn.BatchNorm2d(channels[0]))

        self.conv2ds: nn.ModuleList = nn.ModuleList(conv2ds)
        self.batchnorm2ds: nn.ModuleList = nn.ModuleList(batchnorm2ds)

        conv1ds: List[nn.Conv1d] = []
        batchnorm1ds: List[nn.BatchNorm1d] = []
        kernel_size: int = 5

        while input_freq_height > 4:
            input_freq_height -= kernel_size - 1
            padding: int = 0
            if input_freq_height % 2 == 1:
                input_freq_height += 1
                padding = 1
            conv1ds.append(nn.Conv1d(channels[0], channels[0], kernel_size=kernel_size, padding=padding))
            input_freq_height //= 2
            batchnorm1ds.append(nn.BatchNorm1d(channels[0]))

        self.conv1ds: nn.ModuleList = nn.ModuleList(conv1ds)
        self.batchnorm1ds: nn.ModuleList = nn.ModuleList(batchnorm1ds)
        self.flatten_features: int = input_freq_height * channels[0]
        self.fc1: nn.Linear = nn.Linear(self.flatten_features, 50)
        self.fc2: nn.Linear = nn.Linear(50, 25)

        self.fc_output: nn.Linear = nn.Linear(25, (consts.TET + 1) * len(OutputKeyEnum.__members__))

        if logger is not None:
            logger.info(repr(self))

    def forward(self, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent: torch.Tensor = self.encode(target)
        output: torch.Tensor = self.decode(latent)
        return output, latent
        # return OutputTensor(output), latent

    def encode(self, target: torch.Tensor) -> torch.Tensor:

        output: torch.Tensor = target

        # noinspection PyTypeChecker
        for conv2d, batchnorm2d in zip(self.conv2ds, self.batchnorm2ds):
            output = batchnorm2d(relu(conv2d(output)))
            # Dropout

        output = output.squeeze(dim=2) + target[0, 0, (self.input_time_width - 1) // 2]

        # noinspection PyTypeChecker
        for conv1d, batchnorm1d in zip(self.conv1ds, self.batchnorm1ds):
            output = max_pool1d(batchnorm1d(relu(conv1d(output))), 2)
            # Dropout

        output = output.view(-1, self.flatten_features)
        # output = output.view(output.shape[0], self.flatten_features)
        output = relu(self.fc1(output))
        output = relu(self.fc2(output))
        return output

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        output = self.fc_output(latent)
        output = output.reshape(output.shape[0], -1, consts.TET + 1)
        return output


def get_accuracy(output: torch.Tensor, answer: torch.Tensor) -> float:
    output_argmax: torch.Tensor = torch.argmax(output, dim=2)
    correct: torch.Tensor = (output_argmax == answer)
    correct_all: torch.Tensor = torch.all(correct, dim=1)
    return correct_all.sum().item() / answer.shape[0]

