from abc import ABC, abstractmethod
from typing import List, Type, Union, Tuple

from fsspark.fs.constants import (ML_METHODS, UNIVARIATE_METHODS,
                                  MULTIVARIATE_METHODS)
from fsspark.fs.core import FSDataFrame
from fsspark.fs.ml import MLCVModel
from fsspark.fs.multivariate import multivariate_filter
from fsspark.fs.univariate import univariate_filter


class FSMethod(ABC):
    """
    A general class for feature selection methods.
    """

    valid_methods: Tuple[str]

    def __init__(self,
                 fs_method,
                 **kwargs):
        """
        Initialize the feature selection method with the specified parameters.
        
        :param fs_method: The feature selection method to be used.
        :param kwargs: Additional keyword arguments for the feature selection method.
        """
        self.fs_method = fs_method
        self.kwargs = kwargs
        self.validate_method(fs_method)

    @property
    def valid_methods(self):
        """
        Get the valid methods for feature selection.

        :return: A tuple of valid methods.
        """
        return tuple(self.valid_methods)

    @abstractmethod
    def validate_method(self, fs_method: str):
        """
        Abstract method to validate the feature selection method.

        :param fs_method: The feature selection method to be validated.
        """
        pass

    @abstractmethod
    def select_features(self, fsdf: FSDataFrame):
        """
        Abstract method to select features using the feature selection method.

        :param fsdf: The data frame on which feature selection is to be performed.
        """
        pass

    @abstractmethod
    def validate_params(self, **kwargs):
        """
        Abstract method to validate the parameters for the feature selection method.

        :param kwargs: The parameters to be validated.
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


class FSUnivariate(FSMethod):
    """
    A class for univariate feature selection methods.

    Attributes:
        fs_method (str): The univariate method to be used for feature selection.
        kwargs (dict): Additional keyword arguments for the feature selection method.
    """

    valid_methods = list(UNIVARIATE_METHODS.keys())

    def __init__(self, fs_method: str, **kwargs):
        """
        Initialize the univariate feature selection method with the specified parameters.

        Parameters:
            fs_method: The univariate method to be used for feature selection.
            kwargs: Additional keyword arguments for the feature selection method.
        """

        super().__init__(fs_method, **kwargs)
        self.validate_method(fs_method)

    def validate_method(self, fs_method: str):
        """
        Validate the univariate method.

        Parameters:
            fs_method: The univariate method to be validated.
        """

        if fs_method not in self.valid_methods:
            raise InvalidMethodError(
                f"Invalid univariate method: {fs_method}. "
                f"Accepted methods are {', '.join(self.valid_methods)}"
            )

    def validate_params(self, **kwargs):
        """
        Validate the parameters for the univariate method.

        Parameters:
            kwargs: The parameters to be validated.
        """
        # Additional validation is done directly in the underlying feature selection method
        pass

    def select_features(self, fsdf) -> FSDataFrame:
        """
        Select features using the specified univariate method.

        Parameters:
            fsdf: The data frame on which feature selection is to be performed.

        Returns:
            The selected features.
        """

        return univariate_filter(
            fsdf, univariate_method=self.fs_method, **self.kwargs
        )

    def __str__(self):
        return f"FSUnivariate(method={self.fs_method}, kwargs={self.kwargs})"

    def __repr__(self):
        return self.__str__()


class FSMultivariate(FSMethod):
    """
    The FSMultivariate class is a subclass of the FSMethod class and is used for multivariate
    feature selection methods. It provides a way to select features using different multivariate methods such as
    multivariate correlation and variance.

    Example Usage
    -------------
    # Create an instance of FSMultivariate with multivariate_method='m_corr'
    fs_multivariate = FSMultivariate(multivariate_method='m_corr')

    # Select features using the multivariate method
    selected_features = fs_multivariate.select_features(fsdf)
    """

    valid_methods = list(MULTIVARIATE_METHODS.keys())

    def __init__(self, fs_method: str, **kwargs):
        """
        Initialize the multivariate feature selection method with the specified parameters.

        Parameters:
            fsdf: The data frame on which feature selection is to be performed.
            fs_method: The multivariate method to be used for feature selection.
            kwargs: Additional keyword arguments for the feature selection method.
        """

        super().__init__(fs_method, **kwargs)
        self.validate_method(fs_method)

    def validate_method(self, multivariate_method: str):
        """
        Validate the multivariate method.

        Parameters:
            multivariate_method: The multivariate method to be validated.
        """

        if multivariate_method not in self.valid_methods:
            raise InvalidMethodError(
                f"Invalid multivariate method: "
                f"{multivariate_method}. Accepted methods are {', '.join(self.valid_methods)}"
            )

    def validate_params(self, **kwargs):
        """
        Validate the parameters for the multivariate method.

        Parameters:
            kwargs: The parameters to be validated.
        """
        # Additional validation is done directly in the underlying feature selection method
        pass

    def select_features(self, fsdf: FSDataFrame):
        """
        Select features using the specified multivariate method.
        """

        return multivariate_filter(
            fsdf, multivariate_method=self.fs_method, **self.kwargs
        )

    def __str__(self):
        return f"FSMultivariate(multivariate_method={self.fs_method}, kwargs={self.kwargs})"

    def __repr__(self):
        return self.__str__()


class FSMLMethod(FSMethod):
    """
    A class for machine learning feature selection methods.

    Attributes:
        fs_method (str): The machine learning method to be used for feature selection.
        kwargs (dict): Additional keyword arguments for the feature selection method.
    """

    valid_methods = list(ML_METHODS.keys())
    _ml_model: MLCVModel = None

    def __init__(self,
                 fs_method: str,
                 rfe: bool = False,
                 rfe_iterations: int = 3,
                 percent_to_keep: float = 0.90,
                 **kwargs):
        """
        Initialize the machine learning feature selection method with the specified parameters.

        Parameters:
            fs_method: The machine learning method to be used for feature selection.
            kwargs: Additional keyword arguments for the feature selection method.
        """

        super().__init__(fs_method, **kwargs)
        self.validate_method(fs_method)

        # set the estimator, grid and cv parameters (or none if not provided)
        self.estimator_params = kwargs.get('estimator_params', None)  # estimator parameters
        self.grid_params = kwargs.get('grid_params', None)  # grid parameters
        self.cv_params = kwargs.get('cv_params', None)  # cross-validation parameters

        # set the machine learning model
        self._ml_model = self._set_ml_model()

        # parameters to control the recursive feature elimination process (rfe)
        self.rfe = rfe
        self.percent_to_keep = percent_to_keep
        self.rfe_iterations = rfe_iterations

    def validate_method(self, fs_method: str):
        """
        Validate the machine learning method.

        Parameters:
            fs_method: The machine learning method to be validated.
        """

        if fs_method not in self.valid_methods:
            raise InvalidMethodError(
                f"Invalid machine learning method: {fs_method}. Accepted methods are {', '.join(self.valid_methods)}"
            )

    def validate_params(self, **kwargs):
        """
        Validate the parameters for the machine learning method.

        Parameters:
            kwargs: The parameters to be validated.
        """
        # Additional validation is done directly in the underlying feature selection method
        pass

    def _set_ml_model(self):
        """
        Select the machine learning model to be used for feature selection.

        Returns:
            The machine learning model.
        """

        model_type = self.fs_method

        self._ml_model = MLCVModel.create_model(
            model_type=model_type,
            estimator_params=self.estimator_params,
            grid_params=self.grid_params,
            cv_params=self.cv_params
        )

        return self._ml_model

    def select_features(self, fsdf: FSDataFrame) -> FSDataFrame:
        """
        Select features using the specified machine learning method.

        Parameters:
            fsdf: The data frame on which feature selection is to be performed.

        Returns:
            FSDataFrame: The data frame with selected features.
        """

        if fsdf is None or fsdf.count_features() == 0 or fsdf.count_instances() == 0:
            raise ValueError("The data frame is empty or does not contain any features.")

        # ml_model = self._ml_model
        performance_metrics = []

        def _fit_and_filter(df: FSDataFrame) -> FSDataFrame:

            # fit the model
            self._ml_model.fit(df)

            # get feature scores
            feature_scores = self._ml_model.get_feature_scores()

            # get feature based on the (percentile) threshold provided
            # expected a dataframe sorted by scores in descending order
            selected_features = feature_scores.iloc[
                                :int(self.percent_to_keep * len(feature_scores))
                                ]['feature_index']

            # get accuracy of the model
            accuracy = self._ml_model.get_accuracy()
            performance_metrics.append(accuracy)
            return df.filter_features_by_index(selected_features, keep=True)

        # Recursive feature elimination
        if self.rfe:
            for iteration in range(self.rfe_iterations):
                print(f"RFE: running {iteration + 1} of {self.rfe_iterations} iterations...")
                fsdf = _fit_and_filter(fsdf)
            # print the performance metrics
            print(f"Performance metrics (accuracy): {performance_metrics}")
            return fsdf
        else:
            fsdf = _fit_and_filter(fsdf)
            # print the performance metrics
            print(f"Performance metrics (accuracy): {performance_metrics}")
            return fsdf

    def __str__(self):
        return f"FSMLMethod(method={self.fs_method}, kwargs={self.kwargs})"

    def __repr__(self):
        return self.__str__()


class FSPipeline:
    """
    The FSPipeline class creates a pipeline of feature selection methods. It provides a way to
    chain multiple feature selection methods together to create a pipeline of feature selection methods.

    Example Usage
    -------------
    # Create an instance of FSPipeline with the specified feature selection methods
    fs_pipeline = FSPipeline(fs_methods=[FSUnivariate('anova'), FSMultivariate('m_corr')])

    # Select features using the pipeline
    selected_features = fs_pipeline.select_features(fsdf)
    """

    _valid_methods: List[Type[Union[FSUnivariate, FSMultivariate, FSMLMethod]]] = [FSUnivariate,
                                                                                   FSMultivariate,
                                                                                   FSMLMethod]

    def __init__(self, fs_stages: List[Union[FSUnivariate, FSMultivariate, FSMLMethod]]):
        """
        Initialize the feature selection pipeline with the specified feature selection methods.

        Parameters:
            fs_stages: A list of feature selection methods to be used in the pipeline.
        """

        self.fs_stages = fs_stages
        self.validate_methods()

    def validate_methods(self):
        """
        Validate the feature selection methods in the pipeline.
        """
        # check if the pipeline contains at least one feature selection method
        if len(self.fs_stages) == 0:
            raise ValueError("The pipeline must contain at least one feature selection method.")

        # check if the feature selection methods are valid
        if not all(isinstance(method, tuple(self._valid_methods)) for method in self.fs_stages):
            raise InvalidMethodError(f"Invalid feature selection method. "
                                     f"Accepted methods are {', '.join([str(m) for m in self._valid_methods])}")

        # check if only one ML method is used in the pipeline
        ml_methods = [method for method in self.fs_stages if isinstance(method, FSMLMethod)]
        if len(ml_methods) > 1:
            raise ValueError("Only one ML method is allowed in the pipeline.")

    def run(self, fsdf: FSDataFrame):
        """
        Run the feature selection pipeline on the specified data frame.

        Parameters:
            fsdf: The data frame on which the feature selection pipeline is to be run.

        Returns:
            The selected features.
        """

        # apply each feature selection method in the pipeline
        # print the stage of the pipeline being executed
        n_stages = len(self.fs_stages)
        fsdf_tmp = fsdf
        for i, method in enumerate(self.fs_stages):
            print(f"Running stage {i + 1} of {n_stages} of the feature selection pipeline: {method}")
            fsdf_tmp = method.select_features(fsdf_tmp)

        return fsdf_tmp

    def __str__(self):
        return f"FSPipeline(fs_methods={self.fs_stages})"

    def __repr__(self):
        return self.__str__()


class InvalidMethodError(Exception):
    """
    Error raised when an invalid feature selection method is used.
    """

    def __init__(self, message):
        super().__init__(message)
