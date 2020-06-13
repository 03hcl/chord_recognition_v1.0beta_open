from PyQt5.QtWidgets import QApplication, QMainWindow

from pyqtgraph import GraphicsLayout, GraphicsLayoutWidget, GraphicsView, GraphicsWindow    # , PlotItem
# from pyqtgraph.Qt import QtGui
# import numpy as np
import pyqtgraph as pg

from pyqtgraph.examples import run

from ...src.utils.music import *
from ...src.utils.visualizer import *
from ...src.utils.visualizer.graphlibs import matplotlib as mpl, pyqtgraph as pqg


def main_graphics_window():

    view: GraphicsView = GraphicsWindow()

    fig: GraphicsLayout = view.centralWidget
    axes = fig.addPlot()

    view.setWindowTitle("view [Test] Window Title")
    print(axes)

    QApplication.instance().exec_()


def main_graphics_layout_widget():

    app = QApplication([])
    view: GraphicsView = GraphicsLayoutWidget()

    fig: GraphicsLayout = view.centralWidget
    axes = fig.addPlot()

    win = QMainWindow()
    win.setCentralWidget(view)
    win.show()
    win.setWindowTitle("win [Test] Window Title")
    print(axes)

    app.exec_()


def main_graphlibs_pyqtgraph():

    # print("open(use with) -> add -> clear -> show -> close")
    print("open(use with) -> setting -> show (-> close)")
    with pqg.Figure(grid=(1, 4)) as fig:
        fig.window_title("[Test] Window Title")
        x = np.random.normal(size=100)
        y = np.random.normal(size=100) + 5
        # x = np.linspace(-100, 100, 1000)
        # y = np.sin(x) / x
        ax = fig.axes((0, 0))
        ax.title("[Test] title")

        # ax.legend(True)
        # ax.plot(x=x, y=y, label="test")
        ax.plot(x, y, "test")
        ax.plot(x, y, "test2")
        # ax.plot(x, y, name="test")
        # ax.plot(x, y, "test", label="collision")
        ax.legend(True)

        ax.color_bar()

        axis_x = ax.axis_x()
        axis_x.label("[Test] axis_x label")
        axis_x.scale_line(LineStyle.Solid)
        # # axis_x.scale(Scale.Log)
        axis_x.range((-5, 1))
        axis_x.range((-np.inf, 1))

        # ticks = [-1, 0, 0.2, 0.5, 1]
        # axis_x._raw.setTicks(ticks)
        axis_x.ticks(np.array([-1, 0, 0.2, 0.5, 1]))
        axis_x.tick_labels(np.array(["minus one", "zero", ".2", ".5", "one"]))
        axis_x.minor_ticks(np.array([-0.5, 0.75]))

        axis_y = ax.axis_y()
        axis_y.label("[Test] axis_y label")
        axis_y.scale_line(LineStyle.Solid)
        # axis_y.scale(Scale.Log)

        ax = fig.axes((0, 1))
        ax.scatter(x=x, y=y)
        ax = fig.axes((0, 2))
        ax.line(x=x, y=y)
        r = ax.axis_x().range((-5, None))
        print(r)
        ax = fig.axes((0, 3))
        # ax.histogram(data=y)
        ax.histogram(data=y, data_range=(-5, None))

        # ax = fig.axes((1, 0))
        # ax.bar(x=np.arange(100), y=y)
        # ax = fig.axes((1, 1), (1, 2))
        # ax.stacked_bar(x=np.arange(100), y=abs(np.random.normal(size=(2, 100))))
        # ax.stacked_bar(x=np.arange(100), y=np.arange(200).reshape(2, 100))
        # ax = fig.axes((1, 3))
        # ax.bar(x=np.arange(100), y=y, orientation=Orientation.Horizontal)

        # ax = fig.axes((2, 0))
        # ax.plot(x=x, y=y, marker_style=MarkerStyle.Dot)
        # ax = fig.axes((2, 1))
        # ax.plot(x=x, y=y, marker_style=MarkerStyle.Pixel)
        # ax = fig.axes((2, 2))
        # ax.plot(x=x, y=y, marker_style=MarkerStyle.Circle)
        # ax = fig.axes((2, 3))
        # ax.plot(x=x, y=y, marker_style=MarkerStyle.Square)
        # ax = fig.axes((3, 0))
        # ax.plot(x=x, y=y, marker_style=MarkerStyle.TriangleDown)
        # ax = fig.axes((3, 1))
        # ax.plot(x=x, y=y, marker_style=MarkerStyle.DiamondThin)
        # ax = fig.axes((3, 2))
        # ax.plot(x=x, y=y, marker_style=MarkerStyle.Plus)
        # ax = fig.axes((3, 3))
        # ax.plot(x=x, y=y, marker_style=MarkerStyle.Nothing)

        # ax = fig.axes((0, 0))
        # ax._raw.addPlot()
        # ax = fig.axes((0, 1))
        # ax._raw.addPlot()
        # fig.clear()
        # ax = fig.axes((0, 0))
        # ax._raw.addPlot()

        # f = fig._raw
        # f.addPlot(row=0, col=2)
        # f.addLayout(row=1, col=0, rowspan=3, colspan=3)
        # f0 = f.addLayout(row=2, col=3)
        # f0.addPlot()
        # f.addPlot(row=3, col=1)
        # fig.clear()

        fig.save_as_png("./results", "test_pyqtgraph")

        fig.show()

    # with Reader("./data/raw/20170205.wav") as stereo_reader:
    #     stereo: Stereo = wave_to_signal(*stereo_reader.read_all())
    # mono: Monaural = stereo.monauralize()
    # vqt_data = mono.vqt(stride=2000)

    # data = np.abs(vqt[0].T)
    # with Figure() as fig:
    #     fig.axes((0, 0)).heatmap(data=np.abs(vqt[0].T))
    #     fig.show()
    #
    # visualize_vqt(mono, vqt, seconds, freqs, graph_library=Figure)

    # visualize_vqt(stereo, vqt, seconds, freqs, graph_library=Figure,
    #               directory_path="./results", file_name="vqt")

    # aspect: float = spectrogram._aspect(9 / 16, vqt.shape)
    # ticks: np.ndarray = np.arange(0, len(frequencies))
    # pitches: np.ndarray = Pitch.from_frequency(frequencies)
    # labels: np.ndarray = np.frompyfunc(lambda x: x.show_note_details(), 1, 1)(pitches)
    #
    # for i, ch in signal.keys():
    #     data: np.ndarray = np.abs(vqt[i].T)
    #     data = np.log(data) if is_logscale else data
    #     with graph_library() as graph:
    #         ax: Axes = graph.axes()
    #         ax.heatmap(data, _data_range(is_logscale), origin="lower", aspect=aspect)
    #         _set_axis_ticks(ax.axis_x(), seconds, major_tick_interval, "Time (s)")
    #         y_axis: Axis = ax.axis_y()
    #         y_axis.ticks(ticks[::b])
    #         y_axis.tick_labels(labels[::b])
    #         y_axis.label("Pitch")
    #         ax.title(_title(title, ch))
    #         ax.color_bar(True, orientation=Axes.Orientation.Horizontal)
    #         if directory_path:
    #             graph.save_as_png(directory_path, _title(file_name, ch))
    #         graph.window_title(_title(title, ch))
    #         graph.show()

    fig: pqg.Figure = pqg.Figure()

    print("open -> add -> show -> close")
    f = fig.open()
    fig.window_title("window_title [Test] Window Title")
    # f.addPlot(row=0, col=2)
    # f.addLayout(row=1, col=0, rowspan=3, colspan=3)
    # f0 = f.addLayout(row=2, col=3)
    # f0.addPlot()
    # f.addPlot(row=3, col=1)
    fl = f.addLayout(row=0, col=0)
    fr = f.addLayout(row=0, col=1)
    fl.addPlot()
    fr.addPlot()
    fig.show()
    fig.close()

    print("open -> show -> close")
    fig.open()
    fig.show()
    fig.close()


def minimum_main2():

    x2 = np.linspace(-100, 100, 1000)
    data2 = np.sin(x2) / x2

    with pqg.Figure(grid=(1, 2)) as fig:
        ax = fig.axes()
        ax.line(x=x2, y=data2 + 1)
        ax.axis_x().range((-50, 30))
        fig.show()

    view: GraphicsView = GraphicsWindow()
    view.setWindowTitle("view [Test] Window Title")

    win: GraphicsLayout = view.centralWidget

    p9 = win.addPlot(title="Zoom on selected region")
    p9.plot(data2)
    p9.setXRange(5, 205, padding=0)

    print("execute...")
    from PyQt5.QtWidgets import QApplication    # , QMainWindow
    QApplication.instance().exec_()


def minimum_main():
    # run()

    # app = QtGui.QApplication([])
    # win = QtGui.QMainWindow()
    # view = GraphicsLayoutWidget()
    # win.setCentralWidget(view)
    # win.show()
    # win.setWindowTitle("win [Test] Window Title")

    view: GraphicsView = GraphicsWindow()
    view.setWindowTitle("view [Test] Window Title")

    fig: GraphicsLayout = view.centralWidget

    axes = fig.addPlot()
    n = 300
    s1 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
    pos = np.random.normal(size=(2, n), scale=1e-5)
    spots = [{'pos': pos[:, i], 'data': 1} for i in range(n)] + [{'pos': [0, 0], 'data': 1}]
    s1.addPoints(spots)
    axes.addItem(s1)

    # p = PlotItem()
    # p.titleLabel = "title test"
    # p.addLine([1, 2, 3], [4, 5, 6])
    # fig.addItem(p)
    # fig.addItem(p, row=1, col=1)

    print("execute...")

    from PyQt5.QtWidgets import QApplication    # , QMainWindow
    QApplication.instance().exec_()


# noinspection PyUnusedLocal
def main_vqt(lib_figure):
    with Reader("./data/raw/20170205.wav") as stereo_reader:
        stereo: Stereo = wave_to_signal(*stereo_reader.read_all())
    mono: Monaural = stereo.monauralize()
    # noinspection PyUnusedLocal
    mono_vqt = mono.vqt(stride=2000)
    # visualize_vqt(mono, mono_vqt, orientation=Axes.Orientation.Vertical, is_logscale=True, graph_library=lib_figure)
    # visualize_vqt(mono, mono_vqt, orientation=Axes.Orientation.Vertical, is_logscale=False, graph_library=lib_figure)


def main():
    run()


if __name__ == '__main__':

    main_vqt(mpl.Figure)
    # main_vqt(pqg.Figure)
    # main_graphlibs_pyqtgraph()

    # main_graphics_window()
    # main_graphics_layout_widget()

    # minimum_main()
    # minimum_main2()

    # main()
