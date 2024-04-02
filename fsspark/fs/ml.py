"""

A set of pre-defined ML algorithms wrapped with cross-validation approach
for feature selection (e.g., rank by feature importance) and prediction.

"""

from typing import List, Any, Dict, Optional, Union

import pandas as pd
from pyspark.ml import Estimator, Model
from pyspark.ml.classification import (RandomForestClassificationModel,
                                       LinearSVCModel,
                                       RandomForestClassifier,
                                       LinearSVC)
from pyspark.ml.evaluation import (Evaluator,
                                   BinaryClassificationEvaluator,
                                   MulticlassClassificationEvaluator,
                                   RegressionEvaluator)
from pyspark.ml.regression import RandomForestRegressionModel, RandomForestRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel, Param

from fsspark.fs.constants import RF_BINARY, RF_MULTILABEL, RF_REGRESSION, LSVC_BINARY, ML_METHODS
from fsspark.fs.core import FSDataFrame

ESTIMATORS_CLASSES = [RandomForestClassifier, RandomForestRegressionModel, LinearSVC]
EVALUATORS_CLASSES = [BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator]


# Define an abstract class that allow to create a factory of models
# with the same interface
# This class allows to perform the following operations:
# - Define an Estimator
# - Define an Evaluator
# - Define a grid of parameters (model tuning)
# - Define a cross-validator (model fitting)
class MLCVModel:
    """
    A factory class for creating various machine learning models with Spark MLlib.
    ML model are created using a cross-validator approach for hyperparameter tuning.
    """

    _cross_validator: CrossValidator = None
    _fitted_cv_model: CrossValidatorModel = None
    _best_model: Model = None
    _fsdf: FSDataFrame = None

    def __init__(self,
                 estimator:  Union[RandomForestClassifier |
                                   RandomForestRegressionModel |
                                   LinearSVC],
                 evaluator: Union[BinaryClassificationEvaluator |
                                  MulticlassClassificationEvaluator |
                                  RegressionEvaluator],
                 estimator_params: Optional[Dict[str, Any]] = None,
                 grid_params: Optional[Dict[str, List[Any]]] = None,
                 cv_params: Optional[Dict[str, Any]] = None):
        """
        Initializes the MLModel with optional estimator, evaluator, and parameter specifications.
        """
        self.estimator = estimator
        self.evaluator = evaluator
        self.estimator_params = estimator_params
        self.grid_params = grid_params
        self.cv_params = cv_params

        self._initialize_model()

    def _initialize_model(self):
        # Validate and set estimator parameters
        if self.estimator:
            self._validate_estimator(self.estimator)
            self._validate_estimator_params(self.estimator_params)
            self._set_estimator_params()

        # Validate and evaluator
        if self.evaluator:
            self._validate_evaluator(self.evaluator)

        # Parse and set grid parameters
        if self.grid_params:
            self.grid_params = self._parse_grid_params(self.grid_params)

        # Initialize and set cross-validator parameters
        self._set_cross_validator()

    def _parse_grid_params(self, grid_params: Dict[str, List[Any]]) -> List[Dict[Param, Any]]:
        """
        Parse the grid parameters to create a list of dictionaries.

        :param grid_params: A dictionary containing the parameter names as keys and a list of values as values.
        :return: A list of dictionaries, where each dictionary represents a set of parameter values.
        """
        grid = ParamGridBuilder()
        for param, values in grid_params.items():
            if hasattr(self.estimator, param):
                grid = grid.addGrid(getattr(self.estimator, param), values)
            else:
                raise AttributeError(f"{self.estimator.__class__.__name__} does not have attribute {param}")
        return grid.build()

    def _validate_estimator(self, estimator: Estimator) -> 'MLCVModel':
        """
        Validate the estimator.

        :param estimator: The estimator to validate.
        :return: The validated estimator.
        """
        # check estimator is an instance of ESTIMATORS_CLASSES
        if not isinstance(estimator, tuple(ESTIMATORS_CLASSES)):
            raise ValueError(f"Estimator must be an instance of {ESTIMATORS_CLASSES}")
        return self

    def _validate_evaluator(self, evaluator: Evaluator) -> 'MLCVModel':
        """
        Validate the evaluator.

        :param evaluator: The evaluator to validate.
        :return: The validated evaluator.
        """
        # check evaluator is an instance of EVALUATORS_CLASSES
        if not isinstance(evaluator, tuple(EVALUATORS_CLASSES)):
            raise ValueError(f"Evaluator must be an instance of {EVALUATORS_CLASSES}")
        return self

    def _validate_estimator_params(self, estimator_params: Dict[str, Any]) -> 'MLCVModel':
        """
        Validate the estimator parameters.

        :param estimator_params: A dictionary containing the parameter names as keys and values as values.
        """
        for param, _ in estimator_params.items():
            if self.estimator.hasParam(param):
                pass
            else:
                raise AttributeError(f"{self.estimator.__class__.__name__} does not have attribute {param}")
        return self

    def _set_estimator_params(self) -> 'MLCVModel':
        """
        Set estimator parameters.
        """
        self.estimator = self.estimator.setParams(**self.estimator_params)
        return self

    def _set_cv_params(self, cv_params: Dict[str, Any]) -> 'MLCVModel':
        """
        Parse the cross-validator parameters to create an instance of CrossValidator.

        :param cv_params: A dictionary containing the parameter names as keys and values as values.
        :return: An instance of CrossValidator.
        """

        for param, value in cv_params.items():
            if hasattr(self._cross_validator, param):
                setattr(self._cross_validator, param, value)
            else:
                raise AttributeError(f"{self._cross_validator.__class__.__name__} does not have attribute {param}")
        return self

    def _set_cross_validator(self) -> 'MLCVModel':
        """
        Build the model using the cross-validator.

        :return: The CrossValidator model.
        """
        try:
            self._cross_validator = CrossValidator(
                estimator=self.estimator,
                estimatorParamMaps=self.grid_params,
                evaluator=self.evaluator,
            ).setParams(**self.cv_params)
            return self
        except Exception as e:
            print(f"An error occurred while creating the CrossValidator: {str(e)}")
            # Handle the exception or raise it to be handled by the caller
            raise

    def fit(self, fsdf: FSDataFrame) -> 'MLCVModel':
        """
        Fit the model using the cross-validator.

        :return: The CrossValidatorModel after fitting.
        """
        # Extract the Spark DataFrame and label column name from FSDataFrame
        self._fsdf = fsdf

        if self._cross_validator is None or self.estimator is None or self.evaluator is None:
            raise ValueError("Cross-validator, estimator, or evaluator not set properly.")

        self._fitted_cv_model = self._cross_validator.fit(self._fsdf.get_sdf_vector())
        return self

    def _get_best_model(self) -> Model:
        """
        Get the best model from the fitted CrossValidatorModel.

        :return: The best model.
        """
        if self._fitted_cv_model is None:
            raise ValueError("CrossValidatorModel not fitted. Use fit() to fit the model.")
        self._best_model = self._fitted_cv_model.bestModel
        return self._best_model

    # define a static method that allows to set a ml model based on the model type
    @staticmethod
    def create_model(model_type: str,
                     estimator_params: Dict[str, Any] = None,
                     grid_params: Dict[str, List[Any]] = None,
                     cv_params: Dict[str, Any] = None) -> 'MLCVModel':
        """
        Set a machine learning model based on the model type.

        :param estimator_params:
        :param cv_params:
        :param model_type: The type of model to set.
        :param grid_params: A dictionary containing the parameter names as keys and a list of values as values.

        :return: An instance of MLModel.
        """
        if model_type == RF_BINARY:
            estimator = RandomForestClassifier()
            evaluator = BinaryClassificationEvaluator()
        elif model_type == RF_MULTILABEL:
            estimator = RandomForestClassifier()
            evaluator = MulticlassClassificationEvaluator()
        elif model_type == RF_REGRESSION:
            estimator = RandomForestRegressor()
            evaluator = RegressionEvaluator()
        elif model_type == LSVC_BINARY:
            estimator = LinearSVC()
            evaluator = BinaryClassificationEvaluator()
        else:
            raise ValueError(f"Unsupported model type: {model_type}."
                             f"Supported model types are: {list(ML_METHODS.keys())}")

        ml_method = MLCVModel(
            estimator=estimator,
            evaluator=evaluator,
            estimator_params=estimator_params,
            grid_params=grid_params,
            cv_params=cv_params
        )

        return ml_method

    def get_feature_scores(self) -> pd.DataFrame:

        # TODO: This function should be able to parse all available models.

        indexed_features = self._fsdf.get_features_indexed()
        best_model = self._get_best_model()

        # raise exception if the model is not none
        if best_model is None:
            raise ValueError("No ML model have been fitted. Use fit() to fit the model.")

        df_features = pd.DataFrame(indexed_features.to_numpy(),
                                   columns=["features"])

        if (isinstance(best_model, RandomForestClassificationModel)
                or isinstance(best_model, RandomForestRegressionModel)):

            df_scores = pd.DataFrame(
                data=best_model.featureImportances.toArray(),
                columns=["scores"]
            )

            df_scores = df_scores.reset_index(level=0).rename(columns={"index": "feature_index"})

            # merge the feature scores with the feature names
            df = df_features.merge(
                df_scores, how="right", left_index=True, right_index=True
            )  # index-to-index merging

            # sort the dataframe by scores in descending order
            df = df.sort_values(by="scores", ascending=False)

            # add feature percentile rank to the features_scores dataframe
            df['percentile_rank'] = df['scores'].rank(pct=True)

            return df

        else:
            raise ValueError("Unsupported model type. "
                             "Only RandomForestClassificationModel, "
                             "RandomForestRegressionModel, and LinearSVCModel are supported.")

    def get_accuracy(self) -> float:
        """
        Get accuracy from a trained CrossValidatorModel (best model).
        # TODO: This function should be able to parse all available models.

        :return: accuracy
        """

        # get the best model from the fitted cross-validator model
        best_model = self._get_best_model()

        if isinstance(best_model, RandomForestClassificationModel):
            acc = best_model.summary.accuracy
        elif isinstance(best_model, LinearSVCModel):
            acc = best_model.summary().accuracy
        else:
            acc = None
        return acc

    def get_accuracy_on_test_data(self, test_data: FSDataFrame) -> float:
        """
        Get accuracy on test data from a trained CrossValidatorModel (best model).

        :param test_data: The test data as a FSDataFrame object.
        :return: accuracy
        """

        # TODO: This function should be able to parse all available models.

        # get the best model from the fitted cross-validator model
        best_model = self._get_best_model()

        # predict the test data
        predictions = best_model.transform(test_data.get_sdf_vector())

        if isinstance(best_model, RandomForestClassificationModel):
            acc = self.evaluator.evaluate(predictions)
        else:
            acc = None
        return acc
