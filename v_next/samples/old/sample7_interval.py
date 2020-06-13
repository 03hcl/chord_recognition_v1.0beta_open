from ...src.utils.music import *


def main():

    maj3: Interval = Interval.M3
    print(maj3)
    print(maj3.name)
    print(maj3.to_str(4))

    # interval1: Interval =
    print(IntervalInterpreter.to_interval("  augmented  13    "))


if __name__ == '__main__':
    main()
