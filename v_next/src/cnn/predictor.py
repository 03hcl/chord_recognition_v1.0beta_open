from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from ..utils import ConfigBase, UtilLogger
from ..utils.torchhelper import adapt_tensor_to_device, \
    Device, ModelSet, PredictorBase
from ..utils.torchhelper.data_model import DataTensorLike   # , DatasetLike, DataLoaderLike
# from ..utils.torchhelper.data_model import TEST_KEY_STR
# from ..utils.visualizer.graphlibs import Axes, Axis, FigureBase, LineStyle
# from ..utils.visualizer.graphlibs.matplotlib import Figure

from .config import Config
from .dataset import create_test_dataset, create_validation_dataset
from .model import get_accuracy


class Predictor(PredictorBase):

    @classmethod
    def create_dataset(cls, config: Config, *, logger: UtilLogger, **kwargs) -> TensorDataset:
        return create_test_dataset(config, config.test_resource, logger=logger)

    @classmethod
    def create_validation_dataset(cls, config: Config, *, logger: UtilLogger, **kwargs) -> TensorDataset:
        return create_validation_dataset(config, config.test_resource, logger=logger)

    @classmethod
    def create_data_loader(cls, config: Config, dataset: TensorDataset, *, logger: UtilLogger, **kwargs) -> DataLoader:
        return DataLoader(dataset, batch_size=config.train.batch_size, shuffle=False)

    @classmethod
    def predict_for_each_iteration(cls, model_set: ModelSet, data: DataTensorLike, *, logger: UtilLogger,
                                   **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        target: torch.Tensor = adapt_tensor_to_device(data[0], model_set.device)
        answer: torch.Tensor = adapt_tensor_to_device(data[1], model_set.device)
        return model_set.model(target)[0], answer

    @classmethod
    def run_append(cls, config: ConfigBase, model_set: ModelSet,
                   dataset: TensorDataset, validation_dataset: TensorDataset,
                   output: Dict[str, Tuple[torch.Tensor, torch.Tensor]], *, device: Optional[Device] = None,
                   logger: Optional[UtilLogger] = None, **kwargs) -> None:

        for key, (predicted, answer) in output.items():
            accuracy: float = get_accuracy(predicted, answer)
            logger.info("{:<10}: ACC = {:8.6f}".format(key, accuracy))
        # print(model_set.model.batchnorm2ds[0].running_mean)
