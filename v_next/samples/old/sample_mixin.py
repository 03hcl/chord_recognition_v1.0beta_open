class A:

    def __init__(self, a_str: str = "A(default)", *args, **kwargs):
        self.a: str = a_str


class B:

    def __init__(self, b_str: str = "B(default)", *args, **kwargs):
        self.b: str = b_str


class AB(A, B):

    def __init__(self, *args, **kwargs):
        A.__init__(self, *args, **kwargs)
        B.__init__(self, *args, **kwargs)
        self.ab: str = "AB"


def main():

    ab: AB = AB()
    print("ab.a = {}".format(ab.a))
    print("ab.b = {}".format(ab.b))
    print("ab.ab = {}".format(ab.ab))


if __name__ == '__main__':
    main()
