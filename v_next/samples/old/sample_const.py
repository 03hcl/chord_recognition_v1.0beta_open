from typing import Optional


def sample(**kwargs):
    print(str(kwargs["arg"]))


def sample2(**kwargs):
    sample(arg="default", **kwargs)


def main():

    # sample() got multiple values for keyword argument 'arg'
    # sample2(arg="test")

    # from ..src.utils.music import consts
    # print(consts.TET)
    # consts.TET = 100
    # print(consts.TET)

    # from ..src.utils.music.interval.interpreter import DATA
    # print(DATA.PrefixDict)
    # DATA.PrefixDict = {}
    # print(DATA.PrefixDict)

    n: Optional[bool] = None
    if n:
        print("n is True.")         # True
    # if n == False:
    #     print("n is False.")        # False
    # if n is None:
    #     print("n is None.")         # None
    if not n:
        print("not n is True.")     # False, None


if __name__ == '__main__':
    main()
