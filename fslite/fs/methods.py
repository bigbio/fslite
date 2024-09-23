from abc import ABC, abstractmethod
from typing import List, Type, Union, Optional, Dict, Any

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

# class FSPipeline:
#     """
#     The FSPipeline class creates a pipeline of feature selection methods. It provides a way to
#     chain multiple feature selection methods together to create a pipeline of feature selection methods.
#
#     Example Usage
#     -------------
#     # Create an instance of FSPipeline with the specified feature selection methods
#     fs_pipeline = FSPipeline(fs_methods=[FSUnivariate('anova'), FSMultivariate('m_corr')])
#
#     # Select features using the pipeline
#     selected_features = fs_pipeline.select_features(fsdf)
#     """
#
#     _valid_methods: List[Type[Union[FSUnivariate, FSMultivariate, FSMLMethod]]] = [
#         FSUnivariate,
#         FSMultivariate,
#         FSMLMethod,
#     ]
#
#     def __init__(
#         self,
#         df_training: FSDataFrame,
#         df_testing: Optional[FSDataFrame],
#         fs_stages: List[Union[FSUnivariate, FSMultivariate, FSMLMethod]],
#     ):
#         """
#         Initialize the feature selection pipeline with the specified feature selection methods.
#
#         Parameters:
#             df_training: The training data frame on which the feature selection pipeline is to be run.
#             df_testing: The testing data frame on which the ML wrapper method (if any) is to be evaluated.
#             fs_stages: A list of feature selection methods to be used in the pipeline.
#         """
#
#         self.df_training = df_training
#         self.df_testing = df_testing
#         self.fs_stages = fs_stages
#         self.validate_methods()
#
#         self.pipeline_results = {}
#
#     def validate_methods(self):
#         """
#         Validate the feature selection methods in the pipeline.
#         """
#         # check if the pipeline contains at least one feature selection method
#         if len(self.fs_stages) == 0:
#             raise ValueError(
#                 "The pipeline must contain at least one feature selection method."
#             )
#
#         # check if the feature selection methods are valid
#         if not all(
#             isinstance(method, tuple(self._valid_methods)) for method in self.fs_stages
#         ):
#             raise InvalidMethodError(
#                 f"Invalid feature selection method. "
#                 f"Accepted methods are {', '.join([str(m) for m in self._valid_methods])}"
#             )
#
#         # check if only one ML method is used in the pipeline
#         ml_methods = [
#             method for method in self.fs_stages if isinstance(method, FSMLMethod)
#         ]
#         if len(ml_methods) > 1:
#             raise ValueError("Only one ML method is allowed in the pipeline.")
#
#     def run(self) -> Dict[str, Any]:
#         """
#         Run the feature selection pipeline.
#
#         Returns:
#            A dictionary with the results of the feature selection pipeline.
#         """
#
#         # apply each feature selection method in the pipeline sequentially
#         n_stages = len(self.fs_stages)
#         fsdf_tmp = self.df_training
#
#         self.pipeline_results.update(n_stages=n_stages)
#
#         for i, method in enumerate(self.fs_stages):
#             print(
#                 f"Running stage {i + 1} of {n_stages} of the feature selection pipeline: {method}"
#             )
#             if isinstance(method, FSMLMethod):
#
#                 fsdf_tmp = method.select_features(fsdf_tmp)
#
#                 # collect the results during the feature selection process (rfe iterations, feature scores, etc.)
#                 self.pipeline_results.update(rfe_iterations=method.rfe_iterations)
#                 self.pipeline_results.update(feature_scores=method.get_feature_scores())
#                 self.pipeline_results.update(eval_metric=method.get_eval_metric_name())
#                 self.pipeline_results.update(
#                     rfe_training_metric=method.get_eval_metric_on_training_rfe()
#                 )
#                 self.pipeline_results.update(
#                     training_metric=method.get_eval_metric_on_training()
#                 )
#
#                 if self.df_testing is not None:
#
#                     # evaluate the final model on the testing data (if available)
#                     testing_metric = method.get_eval_metric_on_testing(self.df_testing)
#                     self.pipeline_results.update(testing_metric=testing_metric)
#
#             else:
#                 fsdf_tmp = method.select_features(fsdf_tmp)
#
#         self.pipeline_results.update(
#             n_initial_features=self.df_training.count_features()
#         )
#         self.pipeline_results.update(n_selected_features=fsdf_tmp.count_features())
#
#         return self.pipeline_results
#
#     def __str__(self):
#         return f"FSPipeline(fs_methods={self.fs_stages})"
#
#     def __repr__(self):
#         return self.__str__()


class InvalidMethodError(Exception):
    """
    Error raised when an invalid feature selection method is used.
    """

    def __init__(self, message):
        super().__init__(message)
