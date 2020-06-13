from scipy import signal as sig

from .sample1_wave_io import *


# noinspection PyPep8Naming
def main():

    print(int(-0.5))

    data_front: int = 32

    n = 512  # データ数
    dt = 0.01  # サンプリング間隔
    f = 1  # 周波数
    fn = 1 / (2 * dt)  # ナイキスト周波数
    t = np.linspace(1, n, n) * dt - dt
    y = np.sin(2 * np.pi * f * t) + 0.5 * np.random.randn(t.size)

    fp = 2  # 通過域端周波数[Hz]
    fs = 3  # 阻止域端周波数[Hz]
    gpass = 1  # 通過域最大損失量[dB]
    gstop = 40  # 阻止域最小減衰量[dB]
    Wp = fp / fn
    Ws = fs / fn

    N, Wn = sig.buttord(Wp, Ws, gpass, gstop)
    # noinspection PyTupleAssignmentBalance
    b1, a1 = sig.butter(N, Wn, btype="low")
    y1 = sig.filtfilt(b1, a1, y)

    plt.figure()
    plt.plot(t, y)
    plt.plot(t, y1)
    plt.show()

    print("Create Various Wave:")
    sine: Monaural = Monaural.create_sine_wave(440, phase=np.pi / 2, seconds=5, sampling_rate=48000)
    saw: Monaural = Monaural.create_sawtooth_wave(440, phase=np.pi / 2, seconds=5, sampling_rate=48000)
    tri: Monaural = Monaural.create_triangle_wave(440, phase=np.pi / 2, seconds=5, sampling_rate=48000)
    square: Monaural = Monaural.create_square_wave(440, phase=np.pi / 2, seconds=5, sampling_rate=48000)
    print_each_key_and_signal(sine, data_front)
    plt.plot(sine["MONO"][: 400])
    plt.plot(saw["MONO"][: 400])
    plt.plot(tri["MONO"][: 400])
    plt.plot(square["MONO"][: 400])
    plt.show()

    print("writing...")

    write_wave(sine, "test_sine_wave")
    write_wave(saw, "test_sawtooth_wave")
    write_wave(tri, "test_triangle_wave")
    write_wave(square, "test_square_wave")

    print("Finished!")


if __name__ == '__main__':
    main()
