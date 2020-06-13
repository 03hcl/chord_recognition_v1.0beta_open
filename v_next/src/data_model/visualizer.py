import math
import os
from subprocess import PIPE, Popen
from typing import Dict, Sequence, List, Optional, Tuple

import numpy as np

from PIL import Image, ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont

import torch

from v_next.src.utils import UtilLogger
from v_next.src.utils.music import Chord, ChordInterpreter, ChordNotation, ChordWithBass, \
    consts, NoneChord, Note     # , Reader, WaveFormat
from v_next.src.utils.music.chord.exceptions import CannotDecodeChordError
from v_next.src.utils.visualizer.videolibs.opencv import Writer

from v_next.src.data_model import ChordSequence, OutputKeyEnum   # , WavAndLabFilePair


def create_video_from_answer(raw_data_directory: str, wav_file: str, output_directory: str,
                             answer: torch.Tensor, start: int, end: int, stride: float,
                             output_file_name: str = "", *, logger: Optional[UtilLogger]) -> Popen:
    chord_strs: List[str] = [_create_chord_str_from_answer(answer[i]) for i in range(end - start)]
    return _create_video(raw_data_directory=raw_data_directory, wav_file=wav_file, output_directory=output_directory,
                         output_file_name=output_file_name or "demo_from_tensor",
                         fps=1 / stride, chord_strs=chord_strs, start_frame=start, end_frame=end,
                         logger=logger)


def create_video_from_chord_sequence(raw_data_directory: str, wav_file: str, output_directory: str,
                                     sequence: ChordSequence, fps: float, sampling_rate: int,
                                     output_file_name: str = "", *, logger: Optional[UtilLogger]) -> Popen:
    fps = float(fps)
    chord_strs: List[str] = [c.raw_str for c in sequence[::int(sampling_rate / fps)]]
    return _create_video(raw_data_directory=raw_data_directory, wav_file=wav_file, output_directory=output_directory,
                         output_file_name=output_file_name or "demo_from_lab_file",
                         fps=fps, chord_strs=chord_strs,
                         start_frame=math.ceil(sequence.start_index * fps / sampling_rate),
                         end_frame=math.floor(sequence.end_index * fps / sampling_rate),
                         logger=logger)


def _create_video(raw_data_directory: str, wav_file: str, output_directory: str, output_file_name: str,
                  chord_strs: Sequence[str], fps: float, start_frame: int, end_frame: int,
                  *, logger: Optional[UtilLogger]) -> Popen:

    avi_path: str = output_directory + os.sep + output_file_name + os.extsep + "avi"

    if logger:
        logger.info("デモ動画(音声なし)を一時出力します。(path = {})".format(avi_path))

    # region Create a video by using OpenCV

    font: FreeTypeFont = ImageFont.truetype(
        font=str(raw_data_directory + os.sep + "RictyDiscord-Regular.ttf"), size=24)
    frame_size: Tuple[int, int] = (240, 40)

    with Writer(file=avi_path, fourcc="XVID", fps=fps, frame_size=frame_size, is_color=False) as writer:

        for i in range(end_frame):

            image: Image = Image.new(mode="L", size=frame_size, color=224)
            if not start_frame <= i < end_frame:
                writer.write_frame(np.asarray(image))
                continue

            draw: ImageDraw = ImageDraw.Draw(image)
            draw.text(xy=(16, 8), text=chord_strs[i - start_frame], fill=32, font=font)
            writer.write_frame(np.asarray(image))

    # endregion

    ffmpeg_path: str = raw_data_directory + os.sep + "ffmpeg-4.2.2-win64-static" + os.sep + "bin"
    mp4_path: str = output_directory + os.sep + output_file_name + os.extsep + "mp4"

    command: List[str] = [
        "cd", ffmpeg_path, "&&", "dir", "&&",
        "ffmpeg", "-i", avi_path, "-i", wav_file, "-c:v", "copy", "-c:a", "aac",
        # "-map", "0:v:0", "-map", "1:a:0", "-pic_fmt", "yuv420p",
        "-y", mp4_path, "&&",
        "del", avi_path,
    ]

    if logger:
        logger.info("デモ動画(音声つき)を出力します。    (path = {})".format(mp4_path))

    return Popen(command, stdout=PIPE, stderr=PIPE, shell=True)


def _create_chord_str_from_answer(answer_i: torch.Tensor, notation: ChordNotation = ChordNotation.Advanced,
                                  *, unknown_str: str = "[Unknown]"):

    note_dict: Dict[OutputKeyEnum, Optional[Note]] = dict()

    # noinspection PyUnusedLocal
    key: OutputKeyEnum

    for key in OutputKeyEnum.__members__.values():
        value: int = answer_i[int(key)].item()
        note_dict[key] = None if value == consts.TET else Note.from_int(value)

    # noinspection PyUnusedLocal
    chord: Chord
    # noinspection PyUnusedLocal
    chord_str: str

    try:
        if all(n is None for n in note_dict.values()):
            chord = NoneChord
        else:
            chord = Chord(root=note_dict[OutputKeyEnum.Root],
                          third=note_dict[OutputKeyEnum.Third],
                          fifth=note_dict[OutputKeyEnum.Fifth])
        return ChordInterpreter.encode_to_str(
            ChordWithBass(chord=chord, bass=note_dict[OutputKeyEnum.Bass]), notation=notation)
    except CannotDecodeChordError:
        return unknown_str
