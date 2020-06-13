from ...src.utils.music import *
from ...src.utils.visualizer.graphlibs.matplotlib import *

import matplotlib.pyplot as plt


def show_fft(s: Signal, wf: WindowFunction, ws: int, st: int,
             *, show_hist: bool = False, shown_n_window: int = 0, shown_n_data: int = 0) -> np.ndarray:
    cfft = s.fft(ws, wf, stride=st).raw
    fft0: np.ndarray = np.log(np.abs(cfft[0]))
    if show_hist:
        plt.imshow(fft0.T)
        plt.colorbar()
        plt.show()
    if shown_n_window > 0:
        for i in range(shown_n_window):
            if shown_n_data == 0:
                plt.plot(fft0[i])
            else:
                plt.plot(fft0[i, : shown_n_data])
        plt.show()
    return fft0


def main():

    data_front: int = 60
    window_size: int = 4096
    stride: int = 2000

    freq: float = 440
    sampling_rate: int = 48000

    print(int(-0.5))

    tri: Monaural = Monaural.create_triangle_wave(freq, phase=np.pi / 2, seconds=5, sampling_rate=sampling_rate)
    # plt.plot(tri[0][: 1200])
    # plt.show()

    show_fft(tri, WindowFunction.Rectangular, window_size, stride,
             show_hist=True, shown_n_window=5)
    show_fft(tri, WindowFunction.Hamming, window_size, stride,
             show_hist=True, shown_n_window=5)

    sine = Monaural.create_sine_wave(256, phase=np.pi / 2, seconds=5, sampling_rate=sampling_rate)
    show_fft(sine, WindowFunction.Rectangular, window_size, stride,
             shown_n_window=5, shown_n_data=int(2 * window_size * 256 / 48000))
    show_fft(sine, WindowFunction.Hamming, window_size, stride,
             shown_n_window=5, shown_n_data=int(2 * window_size * 256 / 48000))

    sine: Monaural = Monaural.create_sine_wave(freq, phase=np.pi / 2, seconds=5, sampling_rate=sampling_rate)

    show_fft(sine, WindowFunction.Rectangular, window_size, stride,
             shown_n_window=5, shown_n_data=int(2 * window_size * 440 / 48000))
    show_fft(sine, WindowFunction.Hamming, window_size, stride,
             shown_n_window=5, shown_n_data=int(2 * window_size * 440 / 48000))
    sine_fft: np.ndarray = show_fft(sine, WindowFunction.Rectangular, window_size, stride)
    print("sine_fft: max = {}, min = {}".format(np.max(sine_fft), np.min(sine_fft)))

    with Reader("./data/raw/20170205.wav") as wave_reader:
        stereo: Stereo = wave_to_signal(*wave_reader.read_all())

    stereo_front: Signal = stereo[: data_front]
    stereo_strided: np.ndarray = stereo_front.view_moving_window(32, stride=3, pad_width=(0, 0))
    print("stereo_strided")
    print(stereo_strided)

    stereo_fft = stereo_front.fft(32, stride=3).raw
    print("fft")
    print(stereo_fft)
    print("abs")
    print(np.abs(stereo_fft))

    # show_fft(stereo, WindowFunctionType.Rectangular, window_size, stride, show_hist=True)
    show_fft(stereo, WindowFunction.Hamming, window_size, stride, show_hist=True)


if __name__ == '__main__':
    main()
