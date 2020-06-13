# import subprocess
# from subprocess import CompletedProcess, Popen
# import sys
#
# from ...src.utils import get_common_root_directory


def main():

    # sys.path.append("U:\\OneDrive\\Documents\\PyCharm")
    # print(sys.path)
    # from utils import get_common_root_directory

    # print(10 / 3)
    # print(get_common_root_directory(__file__))

    # command = "cd ./data/raw/ffmpeg-4.2.2-win64-static/bin && dir && ffmpeg --help"
    # command = [
    #     "cd", "./data/raw/ffmpeg-4.2.2-win64-static/bin", "&&", "dir", "&&",
    #     "ffmpeg",
    #     "-i",
    #     "U:\\OneDrive\\Documents\\PyCharm\\chord_analyzer\\v_next\\data\\processed\\ConfigCheck\\20170205\\answer.avi",
    #     "-i",
    #     "U:\\OneDrive\\Documents\\PyCharm\\chord_analyzer\\v_next\\data\\raw\\20170205.wav",
    #     "-c:v", "copy", "-c:a", "aac",
    #     "U:\\OneDrive\\Documents\\PyCharm\\chord_analyzer\\v_next\\data\\processed\\ConfigCheck\\20170205\\demo2.mp4",
    # ]
    # process: Popen = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    # # 長い処理を走らせてよい
    # process.wait()

    print(slice(None, None, -1).indices(5))
    print(dict([]))
    print(dict([("1", 1), ("2", 2)]))


if __name__ == '__main__':
    main()
