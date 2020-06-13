from ..utils.exception_base import SourceException


class DataModelException(SourceException):
    def __init__(self, message: str = None, default_message: str = "data_model model error.", *args, **kwargs):
        super(DataModelException, self).__init__(message, default_message, *args, **kwargs)


class WrongChordRangeError(DataModelException):
    pass


class ResourceRootDirectoryError(DataModelException):
    pass
