from abc import ABC, abstractmethod
from typing import List

from fslite.fs.constants import get_fs_method_details
from fslite.fs.fdataframe import FSDataFrame


class FSMethod(ABC):
    """
    Feature selection abtract class, this class defines the basic structure of a feature selection method.
    From this class are derived the specific feature selection methods like FSUnivariate,
    FSMultivariate and FSMLMethod.
    """

    valid_methods: List[str] = []

    def __init__(self, fs_method, **kwargs):
        """
        Initialize the feature selection method with the specified parameters.

        :param fs_method: The feature selection method to be used.
        :param kwargs: Additional keyword arguments for the feature selection method.
        """
        self.fs_method = fs_method
        self.kwargs = kwargs
        self.validate_method(fs_method)

    @abstractmethod
    def validate_method(self, fs_method: str) -> bool:
        """
        Abstract method to validate the feature selection method.

        :param fs_method: The feature selection method to be validated.
        """
        return get_fs_method_details(fs_method) is not None

    @abstractmethod
    def select_features(self, fsdf: FSDataFrame):
        """
        Abstract method to select features using the feature selection method.

        :param fsdf: The data frame on which feature selection is to be performed.
        """
        pass

    def get_params(self):
        """
        Get the parameters for the feature selection method.

        :return: The parameters as a copy of the kwargs dictionary.
        """
        return self.kwargs.copy()

    def set_params(self, **kwargs):
        """
        Set the parameters for the feature selection method.

        :param kwargs: The new parameters to be set.
        """
        self.kwargs.update(kwargs)


class InvalidMethodError(ValueError):
    """
    Error raised when an invalid feature selection method is used.
    """

    def __init__(self, message):
        super().__init__(f"Invalid feature selection method: {message}")


class InvalidDataError(ValueError):
    """
    Error raised when an invalid feature selection method is used.
    """

    def __init__(self, message):
        super().__init__(f"Invalid data frame: {message}")
