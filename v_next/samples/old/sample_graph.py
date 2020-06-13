# import time
# from typing import Type

from ...src.utils.music import *
from ...src.utils.visualizer import *
from ...src.utils.visualizer.graphlibs import Figure, Axes, Axis
# noinspection PyUnresolvedReferences
from ...src.utils.visualizer.graphlibs import matplotlib as mpl, pyqtgraph as pqg


def plot_and_show_graph(graph_library: Type[Figure], data: np.ndarray, freqs: np.ndarray, freq_labels: np.ndarray):

    # fig1, ax1 = plt.subplots()
    # ax1.imshow(X=data)
    # g0: lib_mpl.Figure = graph_library()
    # g0.open()

    # with graph_library() as graph:
    #     ax: Axes = graph.axes()
    #     ax.heatmap(data)
    #     ax.title("Variable Q Transform")
    #     # print(plt.get_fignums())
    #     graph.show()

    with graph_library() as graph:
        ax: Axes = graph.axes()
        # ax.plot(freqs, data[:, 72])
        ax.line(freqs, data[:, 72])
        # ax.bar(freqs, data[:, 72])
        # ax.scatter(freqs, data[:, 72])
        ax.axis_x().scale(Axis.Scale.Log)

        y_axis: Axis = ax.axis_y()
        # y_scale: Axis.Scale = y_axis.scale(Axis.Scale.Linear)
        # y_axis.ticks(np.array([0.1, 0.05, 0.025, 0.0125, 0.00625]))
        y_scale: Axis.Scale = y_axis.scale(Axis.Scale.Log)
        # y_scale: Axis.Scale = y_axis.scale(Axis.Scale.Logit)
        # y_scale: Axis.Scale = y_axis.scale(Axis.Scale.Power, a=1/2)
        # y_axis.clear()

        ax.grid()
        ax.title("Variable Q Transform (t = 3), scale = " + str(y_scale))
        graph.save_as_png("./results", "vqt_t=3_line")
        graph.show()

    with graph_library() as graph:
        ax: Axes = graph.axes()
        # ax.line(freqs, data[:, 72])
        # ax.axis_x().scale(Axis.Scale.Log)
        ax.bar(np.arange(0, 217), data[:, 72], width=1)
        freq_label_shorten = np.array([x.show_note_details() for x in freq_labels[::24]])
        ax.axis_x().ticks(np.arange(0, 217, 24))
        ax.axis_x().tick_labels(freq_label_shorten)
        ax.axis_y().scale(Axis.Scale.Log)
        # y_scale: Axis.Scale = y_axis.scale(Axis.Scale.Linear)
        # y_scale: Axis.Scale = y_axis.scale(Axis.Scale.Log)
        # y_scale: Axis.Scale = y_axis.scale(Axis.Scale.Logit)
        # y_scale: Axis.Scale = y_axis.scale(Axis.Scale.Power, a=1/2)
        # y_axis.clear()

        ax.minor_grid()
        ax.title("Variable Q Transform (t = 3), scale = " + str(y_scale))
        graph.save_as_png("./results", "vqt_t=3_bar")
        graph.show()

    # ax1.imshow(X=data)
    # print(plt.get_fignums())
    # print(g0._figure_number)
    # g0.show()
    # plt.show()


def main():

    window_size: int = 4096
    stride: int = 2000
    # figure_type: Type[Figure] = mpl.Figure
    figure_type: Type[Figure] = pqg.Figure
    # noinspection PyUnusedLocal
    orientation: Axes.Orientation = Axes.Orientation.Vertical

    with Reader("./data/raw/20170205.wav") as stereo_reader:
        stereo: Stereo = wave_to_signal(*stereo_reader.read_all())

    # noinspection PyUnusedLocal
    vqt0 = stereo.vqt(stride=stride)
    # noinspection PyUnusedLocal
    fft0 = stereo.fft(window_size, stride=stride)

    # visualize_vqt(stereo, vqt0, graph_library=figure_type, orientation=orientation,
    #               directory_path="./results", file_name="vqt")
    # visualize_vqt(stereo, vqt0, graph_library=figure_type, orientation=orientation, is_logscale=True,
    #               directory_path="./results", file_name="vqt_log")
    # visualize_fft(stereo, fft0, graph_library=figure_type, orientation=orientation,
    #               directory_path="./results", file_name="fft")
    # visualize_fft(stereo, fft0, graph_library=figure_type, orientation=orientation, is_logscale=True,
    #               directory_path="./results", file_name="fft_log")

    sig: Monaural = stereo.monauralize()
    c_vqt = sig.vqt(stride=stride)
    vqt_raw: np.ndarray = np.abs(c_vqt.raw[0])

    freq_labels: np.ndarray = Pitch.from_frequency(c_vqt.frequencies)
    plot_and_show_graph(figure_type, vqt_raw.T, c_vqt.frequencies, freq_labels)

    # time.sleep(5)
    # print("slept 5 sec")
    print("end")


if __name__ == '__main__':
    main()
