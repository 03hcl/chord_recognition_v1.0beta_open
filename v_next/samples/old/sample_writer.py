from typing import Type

# noinspection PyUnresolvedReferences
import cv2

import numpy as np

from PIL import Image, ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont

from ...src.utils import get_common_root_directory
from ...src.utils.visualizer.videolibs import Writer
from ...src.utils.visualizer.videolibs.opencv import Writer as OpenCVWriter


def main():

    print(10 / 3)
    root_dir = get_common_root_directory(__file__)
    print(root_dir)

    font: FreeTypeFont = ImageFont.truetype(
        font=str(root_dir.joinpath("./data/raw/RictyDiscord-Regular.ttf")), size=24)

    writer_type: Type[Writer] = OpenCVWriter
    # fourcc: int = cv2.VideoWriter_fourcc(*"XVID")
    # fourcc: int = cv2.VideoWriter_fourcc(*"mp4v")

    with writer_type(file=str(root_dir.joinpath("./results/mono.avi")),
                     fourcc="XVID", fps=30.0, frame_size=(60, 40), is_color=False) as writer:
        for i in range(100):
            image: Image = Image.new(mode="L", size=(60, 40), color=224)
            draw: ImageDraw = ImageDraw.Draw(image)
            draw.text(xy=(0, 10), text="{:2d}".format(i), fill=32, font=font)
            frame: np.ndarray = np.asarray(image)
            writer.write_frame(frame)

    with writer_type(file=str(root_dir.joinpath("./results/color.avi")),
                     fourcc="XVID", fps=30.0, frame_size=(60, 40)) as writer:
        for i in range(100):
            image: Image = Image.new(mode="RGB", size=(60, 40), color=(255, 255, 224))
            draw: ImageDraw = ImageDraw.Draw(image)
            draw.text(xy=(0, 10), text="{:2d}".format(i), fill=(32, 32, 0), font=font)
            frame: np.ndarray = np.asarray(image)
            writer.write_frame(frame)


if __name__ == '__main__':
    main()
