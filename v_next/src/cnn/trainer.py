from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

from ..utils import UtilLogger
from ..utils.torchhelper import adapt_tensor_to_device, Device, TrainerBase, ModelSet
from ..utils.torchhelper.data_model import EpochResult, TrainLog, TrainResult
from ..utils.visualizer import visualize_loss, visualize_values_for_each_epoch

# from ..data_model import OutputTensor

from .config import Config
from .dataset import create_train_dataset, create_validation_dataset
from .model import get_accuracy


class Trainer(TrainerBase):

    @classmethod
    def create_dataset(cls, config: Config, *, logger: UtilLogger, **kwargs) -> TensorDataset:
        return create_train_dataset(config, config.train_resource, logger=logger)

    @classmethod
    def create_validation_dataset(cls, config: Config, *, logger: UtilLogger, **kwargs) -> TensorDataset:
        return create_validation_dataset(config, config.test_resource, logger=logger)

    @classmethod
    def create_data_loader(cls, config: Config, dataset: TensorDataset, *, logger: UtilLogger, **kwargs) -> DataLoader:
        return DataLoader(dataset, batch_size=config.train.batch_size, shuffle=False)

    @classmethod
    def train_for_each_iteration(cls, model_set: ModelSet, data: List[torch.Tensor], backpropagate: bool,
                                 *, logger: UtilLogger, **kwargs) -> TrainResult:

        target: torch.Tensor = adapt_tensor_to_device(data[0], model_set.device)
        answer: torch.Tensor = adapt_tensor_to_device(data[1], model_set.device)

        output: torch.Tensor = model_set.model(target)[0]
        loss: torch.Tensor = sum(model_set.criterion(output[:, i], answer[:, i]) for i in range(answer.shape[1]))
        if backpropagate:
            loss.backward()

        return TrainResult(output=output, loss=loss.item(), target=answer)
        # return TrainResult(output=output, loss=loss.item(), target=(target, answer))

    # noinspection PyUnusedLocal
    @classmethod
    def score_for_each_iteration(cls, model_set: ModelSet, data: List[torch.Tensor], train_result: TrainResult,
                                 *, logger: UtilLogger, **kwargs) -> float:
        return get_accuracy(train_result.output, train_result.target)

    @classmethod
    def output_progress(cls, config: Config, epoch: int, model_set: ModelSet,
                        epoch_result_dict: Dict[str, EpochResult],
                        train_keys: Tuple[str, ...], validation_keys: Tuple[str, ...],
                        loss_array: np.ndarray, score_array: Optional[np.ndarray] = None,
                        visualizes_loss_on_logscale: bool = False, *, logger: UtilLogger, **kwargs) -> None:
        loss_str: str = " / ".join("{}: {:10.6f}".format(k, v.loss) for k, v in epoch_result_dict.items())
        score_str: str = " / ".join("{}: {:5.3f}".format(k, v.score) for k, v in epoch_result_dict.items())
        logger.snap_epoch_with_loss(0, epoch, config.train.number_of_epochs, pre_epoch=config.train.pre_epoch or 0,
                                    customized_log_str_format="(" + loss_str + ")",
                                    log_prefix="[Score = (" + score_str + ")]")
        visualize_loss(loss_array, tuple(epoch_result_dict.keys()),
                       directory=config.interim_directory, file_name="loss" + config.get_epoch_str_function(epoch),
                       pre_epoch=config.train.pre_epoch, is_logscale=visualizes_loss_on_logscale)
        visualize_values_for_each_epoch(
            score_array, tuple(epoch_result_dict.keys()),
            directory=config.interim_directory, file_name="score" + config.get_epoch_str_function(epoch),
            pre_epoch=config.train.pre_epoch, y_axis_name="score", is_logscale=False, data_range=(0, 1))
        # print(model_set.model.batchnorm2ds[0].running_mean)

    @classmethod
    def run_append(cls, config: Config, model_set: ModelSet, result_directory: str,
                   dataset: TensorDataset, validation_dataset: TensorDataset, train_log: TrainLog,
                   *, device: Optional[Device] = None, logger: Optional[UtilLogger] = None, **kwargs) -> None:
        visualize_values_for_each_epoch(
            train_log.score_array, train_log.data_keys,
            directory=result_directory, file_name="score",
            pre_epoch=config.train.pre_epoch, y_axis_name="score", is_logscale=False, data_range=(0, 1))
