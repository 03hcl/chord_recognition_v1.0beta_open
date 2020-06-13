from ..utils.exception_base import SourceException


class CNNException(SourceException):
    def __init__(self, message: str = None, default_message: str = "simple_cnn model error.", *args, **kwargs):
        super(CNNException, self).__init__(message, default_message, *args, **kwargs)


class NotOddTimeWidthError(CNNException):
    pass
