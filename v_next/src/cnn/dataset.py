import math
import os
import re
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch import randperm
from torch.utils.data import TensorDataset

from ..utils import FileName, UtilLogger
from ..utils.music import ChordInterpreter, ChordNotation, ChordWithBass, \
    consts, Monaural, NoneChord, NoteLanguage, Reader, Signal, VariableQ, VQT, wave_to_signal

from ..data_model import AnalyzedData, ChordRange, ChordSequence, OutputKeyEnum, WavAndLabFilePair, \
    create_video_from_answer, create_video_from_chord_sequence
from ..data_model.exceptions import WrongChordRangeError

from .config import Config, ResourceConfig, TestResourceConfig, TrainResourceConfig


def save_data_tensor(config: Config, resource: ResourceConfig,
                     chord_notation: ChordNotation = ChordNotation.MIREX,
                     note_language: NoteLanguage = NoteLanguage.MIDI,
                     logger: Optional[UtilLogger] = None) -> int:
    """データを生成します。データ増強前の件数を返します。"""

    if logger is not None:
        logger.info("データを生成します。")

    rows: int = 0
    qs: Dict[int, VariableQ] = {}

    for file in resource.create_wav_and_lab_file_pairs():

        processed_tensor_directory: str = config.processed_data_directory + os.sep + file.key
        if not os.path.isdir(processed_tensor_directory):
            os.makedirs(processed_tensor_directory, exist_ok=True)

        if not (os.path.isfile(processed_tensor_directory + os.sep + config.data_file.get_full_name(prefix="_")) and
                os.path.isfile(processed_tensor_directory + os.sep + config.answer_file.get_full_name(prefix="_"))):
            if logger is not None:
                logger.info("(ファイル名: {}) データを解析してテンソルを保存します。".format(os.path.split(file.lab)[1]))
            analyzed = create_analyzed_data_and_answer(config, file, qs, chord_notation, note_language)
            data: torch.Tensor = analyzed.data
            answer: torch.Tensor = analyzed.answer
            create_video_from_answer(
                config.raw_data_directory, file.wav, processed_tensor_directory,
                answer, analyzed.start_index, analyzed.end_index, config.stride, logger=logger).wait()
            create_video_from_chord_sequence(
                config.raw_data_directory, file.wav, processed_tensor_directory,
                analyzed.chord_sequence, 30, analyzed.sampling_rate, logger=logger).wait()
            rows += analyzed.end_index - analyzed.start_index
            for t, tensors in augment_by_transposition(config, data, answer):
                augmented_tensor_directory: str = processed_tensor_directory + os.sep + _str_t(t)
                os.makedirs(augmented_tensor_directory, exist_ok=True)
                torch.save(tensors[0], augmented_tensor_directory + os.sep + config.data_file.full_name)
                torch.save(tensors[1], augmented_tensor_directory + os.sep + config.answer_file.full_name)
            torch.save(data, processed_tensor_directory + os.sep + config.data_file.get_full_name(prefix="_"))
            torch.save(answer, processed_tensor_directory + os.sep + config.answer_file.get_full_name(prefix="_"))
        else:
            rows += \
                torch.load(processed_tensor_directory + os.sep + config.answer_file.get_full_name(prefix="_")).shape[0]

    return rows


def create_analyzed_data_and_answer(config: Config, file: WavAndLabFilePair, qs: Dict[int, VariableQ],
                                    chord_notation: ChordNotation, note_language: NoteLanguage) -> AnalyzedData:

    with Reader(file.wav) as reader:
        signal: Signal = wave_to_signal(*reader.read_all())

    mono: Monaural = signal.monauralize()

    if mono.sampling_rate not in qs:
        qs[mono.sampling_rate] = mono.create_variable_q(**config.q_parameter)
    # qs.setdefault(mono.sampling_rate, mono.create_variable_q(**config.q_parameter))

    actual_stride: int = int(mono.sampling_rate * config.stride)
    vqt: VQT = mono.vqt(stride=actual_stride, precalculated_vq=qs[mono.sampling_rate])

    time_width: int = config.model_set.model.parameters["input_time_width"]
    data: np.ndarray = vqt.view_moving_window(time_width, pad_width=(time_width // 2, time_width // 2))
    data = np.abs(data)
    data = np.squeeze(data, axis=0)
    data = np.expand_dims(data, axis=1)

    answer: torch.Tensor = torch.empty((data.shape[0], len(OutputKeyEnum.__members__)), dtype=torch.long)

    chord_range_list: List[ChordRange] = []
    # vqt_tensor: torch.Tensor = torch.from_numpy(data).float()
    # print(vqt_tensor.shape)

    with open(file.lab) as f:

        file_start_index: Optional[int] = None
        end: Optional[int] = None

        for line in f:

            m = re.match(r"^([\d.]+)\s+([\d.]+)\s+(.*?)$", line)
            if not m:
                continue

            start: int = round((float(m.group(1)) - file.gap) * mono.sampling_rate)
            start_index: int = max(start // actual_stride, 0)
            if end is None:
                file_start_index = start_index
            elif start != end:
                raise WrongChordRangeError

            end = round((float(m.group(2)) - file.gap) * mono.sampling_rate)
            end_index: int = max(math.ceil(end / actual_stride), 0)
            if end < start:
                raise WrongChordRangeError

            chord: ChordWithBass = \
                ChordInterpreter.decode_from_str(m.group(3), chord_notation, note_language)

            chord_range_list.append(ChordRange(start, end, chord, m.group(3)))

            if chord.chord is NoneChord:
                answer[start_index: end_index].fill_(consts.TET)
            else:
                answer[start_index: end_index, int(OutputKeyEnum.Root)] = int(chord.chord.root)
                # answer[start_index: end_index, int(OutputKeyEnum.Root)] \
                #     = int(chord.chord.root) if not chord.chord.omits_root else consts.TET
                answer[start_index: end_index, int(OutputKeyEnum.Third)] \
                    = chord.chord.note_3rd if chord.chord.interval_3rd else consts.TET
                answer[start_index: end_index, int(OutputKeyEnum.Fifth)] \
                    = chord.chord.note_5th if chord.chord.interval_5th else consts.TET
                answer[start_index: end_index, int(OutputKeyEnum.Bass)] = int(chord.bass)
                # answer[start_index: end_index, int(OutputKeyEnum.Bass)] \
                #     = int(chord.bass) if chord.bass is not None else consts.TET

            if end_index > answer.shape[0]:
                break

    file_end_index: int = end // actual_stride + 1
    if file_end_index > answer.shape[0]:
        file_end_index = answer.shape[0]
    elif end % actual_stride == 0:
        answer[file_end_index - 1] = answer[math.ceil(end / actual_stride) - 1]

    data: torch.Tensor = torch.from_numpy(data).float()[file_start_index: file_end_index]
    answer = answer[file_start_index: file_end_index]

    return AnalyzedData(data, answer, file_start_index, file_end_index,
                        ChordSequence(chord_range_list), mono.sampling_rate)


def augment_by_transposition(config: Config, data: torch.Tensor, *tensors: torch.Tensor) \
        -> Iterator[Tuple[int, List[torch.Tensor]]]:
    def get_diff(_t: int):
        return -round(_t * config.q_parameter["b"] / consts.TET)

    for t in _transpositions():

        augmented: List[torch.Tensor] = []
        diff: int = get_diff(t)

        zeros: torch.Tensor = torch.zeros((*data.shape[:-1], abs(diff)), dtype=data.dtype)

        if diff > 0:
            augmented.append(torch.cat((data[:, :, :, diff:], zeros), dim=3))
        elif diff < 0:
            augmented.append(torch.cat((zeros, data[:, :, :, :diff]), dim=3))
        else:
            augmented.append(data)

        for tensor in tensors:
            if t != 0:
                transposed: torch.Tensor = (tensor + t) % consts.TET
                transposed[tensor == consts.TET] = consts.TET
                augmented.append(transposed)
            else:
                augmented.append(tensor)

        yield t, augmented


def load_dataset_tensors(config: Config, resource: ResourceConfig, t: int, logger: UtilLogger) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    tensor_data: torch.Tensor = torch.empty(())
    tensor_answer: torch.Tensor = torch.empty(())
    for file in resource.create_wav_and_lab_file_pairs():
        if logger is not None:
            logger.info("(ファイル名: {}) データとなるテンソルを読み込みます。".format(os.path.split(file.lab)[1], t))
        tensor_directory: str = config.processed_data_directory + os.sep + file.key + os.sep + _str_t(t)
        tensor_data = _cat_or_copy(
            tensor_data, torch.load(tensor_directory + os.sep + config.data_file.full_name))
        tensor_answer = _cat_or_copy(
            tensor_answer, torch.load(tensor_directory + os.sep + config.answer_file.full_name))
    return tensor_data, tensor_answer


def save_dataset_tensors(config: Config, resource: TrainResourceConfig,
                         chord_notation: ChordNotation, note_language: NoteLanguage,
                         logger: Optional[UtilLogger] = None) -> None:
    if logger is not None:
        logger.info("データセットとなるテンソルを保存します。")

    length: int = save_data_tensor(config, resource, chord_notation, note_language, logger)
    shuffled: torch.Tensor = randperm(length)

    test_length: int = int(length * resource.test_rate)
    validation_length: int = int((length - test_length) * resource.validation_rate)
    train_length: int = length - (test_length + validation_length)

    def save_data(origin_dataset: TensorDataset, indices: torch.Tensor, file_path: str):
        data: torch.Tensor = torch.empty(
            (indices.shape[0], *origin_dataset.tensors[0].shape[1:]), dtype=origin_dataset.tensors[0].dtype)
        answer: torch.Tensor = torch.empty(
            (indices.shape[0], len(OutputKeyEnum.__members__)), dtype=origin_dataset.tensors[1].dtype)
        for i in range(indices.shape[0]):
            index: int = indices[i].item()
            data[i] = origin_dataset[index][0]
            answer[i] = origin_dataset[index][1]
        torch.save(TensorDataset(data, answer), file_path)

    for t in _transpositions():
        if logger is not None:
            logger.info("transpose = {}".format(_str_t(t)))
        dataset: TensorDataset = TensorDataset(*load_dataset_tensors(config, resource, t, logger))
        save_data(dataset, shuffled[: train_length], _get_train_file_name(config, t))
        if t == 0:
            offset: int = train_length
            save_data(dataset, shuffled[offset: offset + validation_length],
                      config.processed_data_directory + os.sep + config.validation_file.full_name)
            offset += validation_length
            save_data(dataset, shuffled[offset: offset + test_length],
                      config.processed_data_directory + os.sep + config.test_file.full_name)


def create_train_dataset(config: Config, resource: TrainResourceConfig,
                         chord_notation: ChordNotation = ChordNotation.MIREX,
                         note_language: NoteLanguage = NoteLanguage.MIDI,
                         logger: Optional[UtilLogger] = None) -> TensorDataset:
    # last_t: Optional[int] = None
    # for t in _transpositions():
    #     last_t = t
    # if not os.path.isfile(_get_train_file_name(config, last_t)):
    save_dataset_tensors(config, resource, chord_notation, note_language, logger)

    tensor_data: torch.Tensor = torch.empty(())
    tensor_answer: torch.Tensor = torch.empty(())

    for t in _transpositions():
        dataset_t: TensorDataset = torch.load(_get_train_file_name(config, t))
        tensor_data = _cat_or_copy(tensor_data, dataset_t.tensors[0])
        tensor_answer = _cat_or_copy(tensor_answer, dataset_t.tensors[1])

    if logger is not None:
        logger.debug("{}".format(tensor_data.shape))

    return TensorDataset(tensor_data, tensor_answer)


def create_test_dataset(config: Config, resource: TestResourceConfig, data_file_path: str = "",
                        logger: Optional[UtilLogger] = None) -> TensorDataset:
    data_file_path = data_file_path or config.processed_data_directory + os.sep + config.test_file.full_name
    if resource.enforced_recreate:
        dataset: TensorDataset = TensorDataset(*load_dataset_tensors(config, resource, 0, logger))
        torch.save(dataset, data_file_path)
        return dataset
    else:
        return torch.load(data_file_path)


def create_validation_dataset(config: Config, resource: TestResourceConfig, data_file_path: str = "",
                              logger: Optional[UtilLogger] = None) -> TensorDataset:
    return create_test_dataset(
        config, resource,
        data_file_path or config.processed_data_directory + os.sep + config.validation_file.full_name, logger)


def _concat(base: torch.Tensor, appended: torch.Tensor, start: int, end: int) -> torch.Tensor:
    return torch.cat((base, appended[start: end + 1]), dim=0)


def _cat_or_copy(base: torch.Tensor, appended: torch.Tensor, dim: int = 0) -> torch.Tensor:
    if len(base.shape):
        return torch.cat((base, appended), dim=dim)
    else:
        return appended


def _load_tensor(directory: str, filename: FileName) -> torch.Tensor:
    return torch.load(directory + os.sep + filename.full_name)


def _transpose_answer(answer: torch.Tensor, d: int) -> torch.Tensor:
    transposed: torch.Tensor = (answer + d) % consts.TET
    transposed[answer == consts.TET] = consts.TET
    return transposed


def _transpositions() -> Iterator[int]:
    for t in range(- consts.TET // 2, consts.TET - consts.TET // 2):
        yield t


def _str_t(t: int) -> str:
    return ("" if t < 0 else "+") + str(t)


def _get_train_file_name(config: Config, t: int) -> str:
    return config.processed_data_directory + os.sep + config.train_file.get_full_name(suffix=_str_t(t))
