import os
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple, Union

from torch import nn
from torch.optim.optimizer import Optimizer

from ..utils import ConfigBase, FileName, ModuleConfig, TrainConfig, \
    iterate_file_path_set
from ..utils.music import ConstantQ

from ..data_model import WavAndLabFilePair
from ..data_model.exceptions import ResourceRootDirectoryError


class ResourceConfig:

    def __init__(self, music_directories: Iterable[str], chord_directories: Iterable[str],
                 sound_gaps: Optional[Union[Iterable[Tuple[str, float]], Dict[str, float]]] = None,
                 *args, **kwargs):
        self.raw_chord_directories: Tuple[str, ...] = tuple(chord_directories)
        self.raw_music_directories: Tuple[str, ...] = tuple(music_directories)
        self.sound_gaps: Dict[str, float] = dict(sound_gaps or {})
        self._root_directory: str = ""

    def _init_root_directory(self, root_directory: str):
        if (not root_directory) or self._root_directory:
            raise ResourceRootDirectoryError
        self._root_directory = root_directory

    def create_wav_and_lab_file_pairs(self) -> Iterator[WavAndLabFilePair]:
        if not self._root_directory:
            raise ResourceRootDirectoryError
        for k, v in iterate_file_path_set(self._root_directory,
                                          self.raw_music_directories, {"lab": self.raw_chord_directories},
                                          key_name="wav", raises_error=False):
            key: str = k.replace(" ", "_")
            yield WavAndLabFilePair(key, v["wav"], v["lab"], self.sound_gaps.get(key, 0))


class TrainResourceConfig(ResourceConfig):
    def __init__(self, test_rate: float = 0.2, validation_rate: float = 0.1,
                 music_directories: Iterable[str] = ("**" + os.sep + "*" + os.extsep + "wav", ),
                 chord_directories: Iterable[str] = (
                         "*" + os.extsep + "lab",
                         "**" + os.sep + "chordlab" + os.sep + "**" + os.sep + "*" + os.extsep + "lab",
                 ),
                 sound_gaps: Optional[Union[Iterable[Tuple[str, float]], Dict[str, float]]] = None,
                 *args, **kwargs):
        super(TrainResourceConfig, self).__init__(
            music_directories=music_directories, chord_directories=chord_directories, sound_gaps=sound_gaps,
            *args, **kwargs)
        self.test_rate: float = test_rate
        self.validation_rate: float = validation_rate


class TestResourceConfig(ResourceConfig):
    def __init__(self, enforced_recreate: bool = True,
                 music_directories: Iterable[str] = ("**" + os.sep + "*" + os.extsep + "wav", ),
                 chord_directories: Iterable[str] = (
                         "*" + os.extsep + "lab",
                         "**" + os.sep + "chordlab" + os.sep + "**" + os.sep + "*" + os.extsep + "lab",
                 ),
                 sound_gaps: Optional[Union[Iterable[Tuple[str, float]], Dict[str, float]]] = None,
                 *args, **kwargs):
        super(TestResourceConfig, self).__init__(
            music_directories=music_directories, chord_directories=chord_directories, sound_gaps=sound_gaps,
            *args, **kwargs)
        self.enforced_recreate: bool = enforced_recreate


class Config(ConfigBase):

    def __init__(self, root_dir: str,
                 model: ModuleConfig[nn.Module], criterion: ModuleConfig[nn.Module], optimizer: ModuleConfig[Optimizer],
                 train: TrainConfig,
                 train_resource: TrainResourceConfig = TrainResourceConfig(),
                 test_resource: TestResourceConfig = TestResourceConfig(),
                 stride: float = 0.1,
                 f_min: float = 440 / 16, f_max: float = 440 * 32, b: int = 24, gamma: Optional[float] = None,
                 window_size_rate: float = 1, zero_threshold: float = 0.01,
                 music_ext: str = "wav", info_file: FileName = FileName("info", "txt"),
                 data_file: FileName = FileName("data", "pth"),
                 answer_file: FileName = FileName("answer", "pth"),
                 *args, **kwargs):

        super(Config, self).__init__(model, criterion, optimizer, train, *args, root_dir=root_dir, **kwargs)

        self.train_resource: TrainResourceConfig = train_resource
        self.test_resource: TestResourceConfig = test_resource

        self.stride: float = stride
        self.q_parameter: Dict[str, Any] = {
            "f_min": f_min, "f_max": f_max, "b": b,
            "gamma": gamma or 24.7 * 9.265 * ((2 ** (1 / b)) - (2 ** (-1 / b))) / 2,
            "window_size_rate": window_size_rate, "zero_threshold": zero_threshold}

        self.music_ext: str = music_ext
        self.info_file: FileName = info_file
        self.data_file: FileName = data_file
        self.answer_file: FileName = answer_file

        # noinspection PyProtectedMember
        self.train_resource._init_root_directory(self.raw_data_directory)
        # noinspection PyProtectedMember
        self.test_resource._init_root_directory(self.raw_data_directory)

        self.train_file: FileName = FileName("train", "pth")
        self.validation_file: FileName = FileName("validation", "pth")
        self.test_file: FileName = FileName("test", "pth")

        # self.train_file_path_pairs: Iterator[WavAndLabFilePair] = train_resource.create_wav_and_lab_file_pairs()
        # self.test_file_path_pairs: Iterator[WavAndLabFilePair] = test_resource.create_wav_and_lab_file_pairs()

        self.model_set.model.parameters["input_freq_height"] = ConstantQ.calculate_n(f_min, f_max, b)
