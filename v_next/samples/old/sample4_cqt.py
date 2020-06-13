from .sample3_fft import *

from ...src.utils.music import *


def show_cqt(s: Signal, wf: WindowFunction = WindowFunction.Hamming, wsr: float = 1, st: int = 4800,
             *, show_hist: bool = True, is_log_scale: bool = False,
             precalculated_cq: Optional[ConstantQ] = None) -> np.ndarray:
    # cq: ConstantQ = precalculated_cq or s.create_constant_q(window_size_rate=wsr, window_type=wf)
    cq: ConstantQ = precalculated_cq or s.create_variable_q(window_size_rate=wsr, window_type=wf, gamma=0)
    c_cqt = s.cqt(precalculated_cq=cq, stride=st).raw
    cqt: np.ndarray = np.abs(c_cqt[0])
    if is_log_scale:
        cqt = np.log(cqt)
    if show_hist:
        plt.imshow(cqt.T, vmin=0, vmax=0.12)
        plt.gca().invert_yaxis()
        plt.yticks(np.arange(cq.n // cq.number_of_bins_in_an_octave * cq.number_of_bins_in_an_octave,
                             0, -cq.number_of_bins_in_an_octave))
        plt.title("CQT")
        plt.colorbar()
        plt.savefig("./results/sample4/cqt.png")
        plt.show()
    return cqt


def show_vqt(s: Signal, wf: WindowFunction = WindowFunction.Hamming, wsr: float = 1, st: int = 4800,
             *, gamma: Optional[float] = None, zero_threshold: float = 0.01,
             show_hist: bool = True, is_log_scale: bool = False,
             precalculated_vq: Optional[ConstantQ] = None) -> np.ndarray:
    b: int = 24
    if gamma is None:
        gamma = 24.7 * 9.265 * ((2 ** (1 / b)) - (2 ** (-1 / b))) / 2
    vq: VariableQ = precalculated_vq or s.create_variable_q(window_size_rate=wsr, window_type=wf, b=b,
                                                            zero_threshold=zero_threshold, gamma=gamma)
    c_vqt = s.vqt(precalculated_vq=vq, stride=st).raw
    vqt0: np.ndarray = np.abs(c_vqt[0])
    if is_log_scale:
        vqt0 = np.log(vqt0)
    if show_hist:
        plt.imshow(vqt0.T, vmin=0, vmax=0.12)
        plt.gca().invert_yaxis()
        plt.yticks(np.arange(vq.n // vq.number_of_bins_in_an_octave * vq.number_of_bins_in_an_octave,
                             -1, -vq.number_of_bins_in_an_octave))
        plt.title("VQT (γ= {:.3f})".format(gamma))
        plt.colorbar()
        plt.savefig("./results/sample4/vqt_gamma_{:.3f}.png".format(gamma))
        plt.show()
    return vqt0


def main():

    # noinspection PyUnusedLocal
    data_front: int = 60
    # noinspection PyUnusedLocal
    window_size: int = 4096
    # noinspection PyUnusedLocal
    stride: int = 2000

    freq: float = 440
    sampling_rate: int = 48000

    vqt_window_size_rate: float = 2
    vqt_gamma: Optional[float] = 3
    vqt_zero_threshold: float = 0.01

    # cq0: ConstantQ = ConstantQ()
    # cq0: ConstantQ = ConstantQ(window_size_rate=0.2)
    # cq0: ConstantQ = ConstantQ(f_min=60, f_max=6000, zero_threshold=0.0054)
    # cq0: ConstantQ = ConstantQ(window_size_rate=window_size_rate, zero_threshold=0.01)

    cq0: ConstantQ = VariableQ(window_size_rate=vqt_window_size_rate, zero_threshold=0.01, gamma=0)
    cq0_k: np.ndarray = np.arange(0, cq0.n)
    cq0_f_k: np.ndarray = np.array([cq0.f_k(k) for k in cq0_k])
    cq0_n_k: np.ndarray = np.array([cq0.n_k(cq0.f_k(k)) for k in cq0_k])

    vq0: VariableQ = VariableQ(window_size_rate=vqt_window_size_rate, zero_threshold=0.01, gamma=vqt_gamma)
    vq0_k: np.ndarray = np.arange(0, vq0.n)
    vq0_f_k: np.ndarray = np.array([vq0.f_k(k) for k in vq0_k])
    vq0_n_k: np.ndarray = np.array([vq0.n_k(vq0.f_k(k)) for k in vq0_k])

    vqd: VariableQ = VariableQ(window_size_rate=vqt_window_size_rate, zero_threshold=0.01, gamma=None)
    vqd_k: np.ndarray = np.arange(0, vqd.n)
    vqd_f_k: np.ndarray = np.array([vqd.f_k(k) for k in vqd_k])
    vqd_n_k: np.ndarray = np.array([vqd.n_k(vqd.f_k(k)) for k in vqd_k])

    plt.plot(cq0_f_k, cq0.sampling_rate / cq0_n_k)
    plt.plot(vq0_f_k, vq0.sampling_rate / vq0_n_k)
    plt.plot(vqd_f_k, vq0.sampling_rate / vqd_n_k)
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.show()

    plt.plot(cq0_k, cq0_n_k, label="CQT")
    plt.plot(vq0_k, vq0_n_k, label="VQT (γ= {:.3f})".format(vq0.gamma))
    plt.plot(vqd_k, vqd_n_k, label="VQT (γ= {:.3f})".format(vqd.gamma))
    plt.legend()
    # plt.gca().set_yscale('log')
    plt.savefig("./results/sample4/window_size.png")
    plt.show()

    # plt.figure(figsize=(12, 12))
    plt.imshow(abs(cq0.kernel).toarray(), aspect=float(cq0.kernel.shape[1]) / cq0.kernel.shape[0])
    plt.gca().invert_yaxis()
    plt.show()
    plt.imshow(abs(vq0.kernel).toarray(), aspect=float(vq0.kernel.shape[1]) / vq0.kernel.shape[0])
    plt.gca().invert_yaxis()
    plt.show()
    plt.imshow(abs(vqd.kernel).toarray(), aspect=float(vqd.kernel.shape[1]) / vqd.kernel.shape[0])
    plt.gca().invert_yaxis()
    plt.show()

    tri: Monaural = Monaural.create_triangle_wave(freq, phase=np.pi / 2, seconds=5, sampling_rate=sampling_rate)
    show_cqt(tri)
    # show_cqt(tri, wsr=vqt_window_size_rate)
    show_vqt(tri, wsr=vqt_window_size_rate, zero_threshold=vqt_zero_threshold)
    show_vqt(tri, wsr=vqt_window_size_rate, gamma=vqt_gamma, zero_threshold=vqt_zero_threshold)

    with Reader("./data/raw/20170205.wav") as wave_reader:
        stereo: Stereo = wave_to_signal(*wave_reader.read_all())

    cqt_mono: np.ndarray = show_cqt(stereo.monauralize())
    # cqt_mono: np.ndarray = show_cqt(stereo.monauralize(), wsr=vqt_window_size_rate)
    vqt_mono: np.ndarray = show_vqt(stereo.monauralize(), wsr=vqt_window_size_rate, zero_threshold=vqt_zero_threshold)
    vqt0_mono: np.ndarray = show_vqt(stereo.monauralize(), wsr=vqt_window_size_rate, gamma=vqt_gamma,
                                     zero_threshold=vqt_zero_threshold)

    print("cqt_mono: max = {}, min = {}".format(np.max(cqt_mono), np.min(cqt_mono)))
    print("vqt_mono: max = {}, min = {}".format(np.max(vqt_mono), np.min(vqt_mono)))
    print("vqt0_mono: max = {}, min = {}".format(np.max(vqt0_mono), np.min(vqt0_mono)))


if __name__ == '__main__':
    main()
