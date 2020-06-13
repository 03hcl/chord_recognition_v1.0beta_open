# from typing import List

import numpy as np
import torch

from ...src.cnn.model import CNN

from ...src.utils import UtilLogger, BlankUtilLogger

# from ..src.utils.transformer import ConstantQ, VariableQ
from ...src.utils.music.signal import VQT, Monaural, Stereo  # , FFT
# from ..src.utils.music.note import *
from ...src.utils.music.wave import wave_to_signal
from ...src.utils.music.wave.stream import Reader


def main():

    with Reader("./data/raw/20170205.wav") as reader:
        stereo: Stereo = wave_to_signal(*reader.read_all())

    mono: Monaural = stereo.monauralize()

    stride: int = 4800
    vqt_window_size = 9

    # vq: VariableQ = mono.create_variable_q()
    # fft = mono.fft(window_size=4096, stride=stride)
    vqt: VQT = mono.vqt(stride=stride)
    vqt_mw: np.ndarray = vqt.view_moving_window(vqt_window_size)
    print(vqt_mw.shape)
    # vqt_mw
    vqt_mw = np.squeeze(vqt_mw, axis=0)
    vqt_mw = np.expand_dims(vqt_mw, axis=1)
    vqt_tensor: torch.Tensor = torch.from_numpy(np.abs(vqt_mw)).float()
    print(vqt_tensor.shape)

    logger: UtilLogger = BlankUtilLogger()

    cnn_test: CNN = CNN(9, vqt.q.n, logger=logger)

    # test_data: torch.Tensor = torch.empty((1, 1, 9, vqt.q.n))
    # test_data[0, 0] = torch.from_numpy(np.abs(vqt.raw[0, : 9, :]))
    # print(test_data.shape)
    test_output = cnn_test.forward(vqt_tensor)

    for t in test_output:
        print(t.shape)
    print("end.")


if __name__ == '__main__':
    main()
