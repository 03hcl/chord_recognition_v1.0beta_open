from enum import auto, IntEnum, unique


@unique
class OutputKeyEnum(IntEnum):
    Bass = 0
    Root = auto()
    Third = auto()
    Fifth = auto()
