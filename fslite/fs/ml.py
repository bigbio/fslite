"""

A set of pre-defined ML algorithms wrapped with cross-validation approach
for feature selection (e.g., rank by feature importance) and prediction.

"""

from fslite.fs.constants import get_fs_ml_methods, is_valid_ml_method
from fslite.fs.fdataframe import FSDataFrame
from fslite.fs.methods import FSMethod, InvalidMethodError


class FSMLMethod(FSMethod):
    """
    A class for machine learning feature selection methods.

    Attributes:
        fs_method (str): The machine learning method to be used for feature selection.
        kwargs (dict): Additional keyword arguments for the feature selection method.
    """

    valid_methods = get_fs_ml_methods()
    _ml_model: MLCVModel = None

    def __init__(
        self,
        fs_method: str,
        rfe: bool = False,
        rfe_iterations: int = 3,
        percent_to_keep: float = 0.90,
        **kwargs,
    ):
        """
        Initialize the machine learning feature selection method with the specified parameters.

        Parameters:
            fs_method: The machine learning method to be used for feature selection.
            kwargs: Additional keyword arguments for the feature selection method.
        """

        super().__init__(fs_method, **kwargs)
        self.validate_method(fs_method)

        # set the estimator, grid and cv parameters (or none if not provided)
        self.estimator_params = kwargs.get(
            "estimator_params", None
        )  # estimator parameters
        self.evaluator_params = kwargs.get(
            "evaluator_params", None
        )  # evaluator parameters
        self.grid_params = kwargs.get("grid_params", None)  # grid parameters
        self.cv_params = kwargs.get("cv_params", None)  # cross-validation parameters

        # set the machine learning model
        self._ml_model = self._set_ml_model()

        # parameters to control the recursive feature elimination process (rfe)
        self.rfe = rfe
        self.percent_to_keep = percent_to_keep
        self.rfe_iterations = rfe_iterations

        # performance metrics
        self.rfe_training_metric: list = (
            []
        )  # performance metrics on training for each rfe iteration
        self.training_metric = None  # performance metrics on training (final model)
        self.testing_metric = None  # performance metrics on testing (final model)

        # feature importance
        self.feature_scores = None

    def validate_method(self, fs_method: str):
        """
        Validate the machine learning method.

        Parameters:
            fs_method: The machine learning method to be validated.
        """

        if not is_valid_ml_method(fs_method):
            raise InvalidMethodError(
                f"Invalid machine learning method: {fs_method}. Accepted methods are {', '.join(self.valid_methods)}"
            )

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
            evaluator_params=self.evaluator_params,
            grid_params=self.grid_params,
            cv_params=self.cv_params,
        )

        return self._ml_model

    def _fit_and_filter(self, df: FSDataFrame) -> FSDataFrame:

        # fit the current machine learning model
        self._ml_model.fit(df)

        # get feature scores
        feature_scores = self._ml_model.get_feature_scores()

        # get feature based on the (percentile) threshold provided
        # expected a dataframe sorted by scores in descending order
        selected_features = feature_scores.iloc[
            : int(self.percent_to_keep * len(feature_scores))
        ]["feature_index"]

        return df.select_features_by_index(selected_features, keep=True)

    def select_features(self, fsdf: FSDataFrame) -> FSDataFrame:
        """
        Select features using the specified machine learning method.

        Parameters:
            fsdf: The data frame on which feature selection is to be performed.

        Returns:
            FSDataFrame: The data frame with selected features.
        """

        if fsdf is None or fsdf.count_features() == 0 or fsdf.count_instances() == 0:
            raise ValueError(
                "The data frame is empty or does not contain any features."
            )

        fsdf = self._fit_and_filter(fsdf)

        # Recursive feature elimination
        if self.rfe:
            for iteration in range(self.rfe_iterations):
                print(
                    f"RFE: running {iteration + 1} of {self.rfe_iterations} iterations..."
                )
                fsdf = self._fit_and_filter(fsdf)
                # collect the performance metrics on training for every rfe iteration
                self.rfe_training_metric.append(
                    self._ml_model.get_eval_metric_on_training()
                )

        # get the final performance metric on training
        self.training_metric = self._ml_model.get_eval_metric_on_training()

        # get the feature scores after feature selection
        self.feature_scores = self._ml_model.get_feature_scores()

        return fsdf

    def get_eval_metric_name(self):
        """
        Get the evaluation metric name.

        Returns:
            The evaluation metric name.
        """

        if self._ml_model is None:
            raise ValueError("No machine learning model is available.")

        return self._ml_model.get_eval_metric_name()

    def get_eval_metric_on_training_rfe(self):
        """
        Get the evaluation metric on the training data for each RFE iteration.

        Returns:
            The evaluation metric on the training data for each RFE iteration.
        """
        if self.rfe_training_metric is None:
            raise ValueError(
                "No training metric is available. Run the select_features method first."
            )
        return self.rfe_training_metric

    def get_eval_metric_on_training(self):
        """
        Get the evaluation metric on the training data.

        Returns:
            The evaluation metric on the training data.
        """
        if self.training_metric is None:
            raise ValueError(
                "No training metric is available. Run the select_features method first."
            )
        return self.training_metric

    def get_eval_metric_on_testing(self, fsdf: FSDataFrame):
        """
        Evaluate the machine learning method on the testing data.

        Parameters:
            fsdf: The testing data frame on which the machine learning method is to be evaluated.

        Returns:
            The evaluation metric on the testing data.
        """

        if fsdf is None or fsdf.count_features() == 0 or fsdf.count_instances() == 0:
            raise ValueError(
                "The testing data frame is empty or does not contain any features."
            )

        # evaluate the model on the testing data
        eval_metric = self._ml_model.get_eval_metric_on_testing(fsdf)
        self.testing_metric = eval_metric

        return eval_metric

    def get_feature_scores(self):
        """
        Get the feature scores after feature selection.

        Returns:
            The feature scores as a pandas DataFrame.
        """

        if self.feature_scores is None:
            raise ValueError(
                "Feature scores are not available. Run the feature selection method first."
            )

        return self.feature_scores

    def __str__(self):
        return f"FSMLMethod(method={self.fs_method}, kwargs={self.kwargs})"

    def __repr__(self):
        return self.__str__()


#
#
# # Define an abstract class that allow to create a factory of models
# # with the same interface
# # This class allows to perform the following operations:
# # - Define an Estimator
# # - Define an Evaluator
# # - Define a grid of parameters (model tuning)
# # - Define a cross-validator (model fitting)
# class MLCVModel:
#     """
#     A factory class for creating various machine learning models with Spark MLlib.
#     ML model are created using a cross-validator approach for hyperparameter tuning.
#     """
#
#     _cross_validator: CrossValidator = None
#     _fitted_cv_model: CrossValidatorModel = None
#     _best_model: Model = None
#     _fsdf: FSDataFrame = None
#
#     def __init__(
#         self,
#         estimator: Union[
#             RandomForestClassifier
#             | RandomForestRegressionModel
#             | LinearSVC
#             | LogisticRegression
#         ],
#         evaluator: Union[
#             BinaryClassificationEvaluator
#             | MulticlassClassificationEvaluator
#             | RegressionEvaluator
#         ],
#         estimator_params: Optional[Dict[str, Any]] = None,
#         evaluator_params: Optional[Dict[str, Any]] = None,
#         grid_params: Optional[Dict[str, List[Any]]] = None,
#         cv_params: Optional[Dict[str, Any]] = None,
#     ):
#         """
#         Initializes the MLModel with optional estimator, evaluator, and parameter specifications.
#         """
#         self.estimator = estimator
#         self.evaluator = evaluator
#         self.estimator_params = estimator_params
#         self.evaluator_params = evaluator_params
#         self.grid_params = grid_params
#         self.cv_params = cv_params
#
#         self._initialize_model()
#
#     def _initialize_model(self):
#         # Validate and set estimator parameters
#         if self.estimator:
#             self._validate_estimator(self.estimator)
#             self._validate_estimator_params(self.estimator_params)
#             self._set_estimator_params()
#
#         # Validate and evaluator
#         if self.evaluator:
#             self._validate_evaluator(self.evaluator)
#             self._validate_evaluator_params(self.evaluator_params)
#             self._set_evaluator_params()
#
#         # Parse and set grid parameters
#         if self.grid_params:
#             self.grid_params = self._parse_grid_params(self.grid_params)
#
#         # Initialize and set cross-validator parameters
#         self._set_cross_validator()
#
#     def _parse_grid_params(
#         self, grid_params: Dict[str, List[Any]]
#     ) -> List[Dict[Param, Any]]:
#         """
#         Parse the grid parameters to create a list of dictionaries.
#
#         :param grid_params: A dictionary containing the parameter names as keys and a list of values as values.
#         :return: A list of dictionaries, where each dictionary represents a set of parameter values.
#         """
#         grid = ParamGridBuilder()
#         for param, values in grid_params.items():
#             if hasattr(self.estimator, param):
#                 grid = grid.addGrid(getattr(self.estimator, param), values)
#             else:
#                 raise AttributeError(
#                     f"{self.estimator.__class__.__name__} does not have attribute {param}"
#                 )
#         return grid.build()
#
#     def _validate_estimator(self, estimator: Estimator) -> "MLCVModel":
#         """
#         Validate the estimator.
#
#         :param estimator: The estimator to validate.
#         :return: The validated estimator.
#         """
#         # check estimator is an instance of ESTIMATORS_CLASSES
#         if not isinstance(estimator, tuple(ESTIMATORS_CLASSES)):
#             raise ValueError(f"Estimator must be an instance of {ESTIMATORS_CLASSES}")
#         return self
#
#     def _validate_evaluator(self, evaluator: Evaluator) -> "MLCVModel":
#         """
#         Validate the evaluator.
#
#         :param evaluator: The evaluator to validate.
#         :return: The validated evaluator.
#         """
#         # check evaluator is an instance of EVALUATORS_CLASSES
#         if not isinstance(evaluator, tuple(EVALUATORS_CLASSES)):
#             raise ValueError(f"Evaluator must be an instance of {EVALUATORS_CLASSES}")
#         return self
#
#     def _validate_estimator_params(self, estimator_params: Dict[str, Any]) -> None:
#         """
#         Validate the estimator parameters.
#
#         :param estimator_params: A dictionary containing the parameter names as keys and values as values.
#         """
#         if estimator_params is None:
#             return
#         for param, _ in estimator_params.items():
#             if not self.estimator.hasParam(param):
#                 raise AttributeError(
#                     f"{self.estimator.__class__.__name__} does not have attribute {param}"
#                 )
#
#     def _validate_evaluator_params(self, evaluator_params: Dict[str, Any]) -> None:
#         """
#         Validate the evaluator parameters.
#
#         :param evaluator_params: A dictionary containing the parameter names as keys and values as values.
#         """
#         if evaluator_params is None:
#             return
#         for param, _ in evaluator_params.items():
#             if not self.evaluator.hasParam(param):
#                 raise AttributeError(
#                     f"{self.evaluator.__class__.__name__} does not have attribute {param}"
#                 )
#
#     def _set_evaluator_params(self) -> "MLCVModel":
#         """
#         Set evaluator parameters.
#         """
#         if self.evaluator_params is not None:
#             self.evaluator = self.evaluator.setParams(**self.evaluator_params)
#         return self
#
#     def _set_estimator_params(self) -> "MLCVModel":
#         """
#         Set estimator parameters.
#         """
#         if self.estimator_params is not None:
#             self.estimator = self.estimator.setParams(**self.estimator_params)
#         return self
#
#     def _set_cv_params(self, cv_params: Dict[str, Any]) -> "MLCVModel":
#         """
#         Parse the cross-validator parameters to create an instance of CrossValidator.
#
#         :param cv_params: A dictionary containing the parameter names as keys and values as values.
#         :return: An instance of CrossValidator.
#         """
#
#         for param, value in cv_params.items():
#             if hasattr(self._cross_validator, param):
#                 setattr(self._cross_validator, param, value)
#             else:
#                 raise AttributeError(
#                     f"{self._cross_validator.__class__.__name__} does not have attribute {param}"
#                 )
#         return self
#
#     def _set_cross_validator(self) -> "MLCVModel":
#         """
#         Build the model using the cross-validator.
#
#         :return: The CrossValidator model.
#         """
#         try:
#             self._cross_validator = CrossValidator(
#                 estimator=self.estimator,
#                 estimatorParamMaps=self.grid_params,
#                 evaluator=self.evaluator,
#             )
#             if self.cv_params is not None:
#                 self._cross_validator = self._cross_validator.setParams(
#                     **self.cv_params
#                 )
#             return self
#         except Exception as e:
#             print(f"An error occurred while creating the CrossValidator: {str(e)}")
#             # Handle the exception or raise it to be handled by the caller
#             raise
#
#     def fit(self, fsdf: FSDataFrame) -> "MLCVModel":
#         """
#         Fit the model using the cross-validator.
#
#         :return: The CrossValidatorModel after fitting.
#         """
#         # Extract the Spark DataFrame and label column name from FSDataFrame
#         self._fsdf = fsdf
#
#         if (
#             self._cross_validator is None
#             or self.estimator is None
#             or self.evaluator is None
#         ):
#             raise ValueError(
#                 "Cross-validator, estimator, or evaluator not set properly."
#             )
#
#         self._fitted_cv_model = self._cross_validator.fit(self._fsdf.get_sdf_vector())
#         return self
#
#     def _get_best_model(self) -> Model:
#         """
#         Get the best model from the fitted CrossValidatorModel.
#
#         :return: The best model.
#         """
#         if self._fitted_cv_model is None:
#             raise ValueError(
#                 "CrossValidatorModel not fitted. Use fit() to fit the model."
#             )
#         self._best_model = self._fitted_cv_model.bestModel
#         return self._best_model
#
#     # define a static method that allows to set a ml model based on the model type
#     @staticmethod
#     def create_model(
#         model_type: str,
#         estimator_params: Dict[str, Any] = None,
#         evaluator_params: Dict[str, Any] = None,
#         grid_params: Dict[str, List[Any]] = None,
#         cv_params: Dict[str, Any] = None,
#     ) -> "MLCVModel":
#         """
#         Set a machine learning model based on the model type.
#
#         :param model_type: The type of model to set.
#         :param estimator_params: Parameters for the estimator.
#         :param evaluator_params: Parameters for the evaluator.
#         :param grid_params: A dictionary containing the parameter names as keys and a list of values as values.
#         :param cv_params: Parameters for the cross-validator.
#
#         :return: An instance of MLModel.
#         """
#         if model_type == RF_BINARY:
#             estimator = RandomForestClassifier()
#             evaluator = BinaryClassificationEvaluator()
#         elif model_type == LSVC_BINARY:
#             estimator = LinearSVC()
#             evaluator = BinaryClassificationEvaluator()
#         elif model_type == RF_MULTILABEL:
#             estimator = RandomForestClassifier()
#             evaluator = MulticlassClassificationEvaluator()
#         elif model_type == LR_MULTILABEL:
#             estimator = LogisticRegression()
#             evaluator = MulticlassClassificationEvaluator()
#         elif model_type == RF_REGRESSION:
#             estimator = RandomForestRegressor()
#             evaluator = RegressionEvaluator()
#         else:
#             raise ValueError(
#                 f"Unsupported model type: {model_type}."
#                 f"Supported model types are: {list(ML_METHODS.keys())}"
#             )
#
#         ml_method = MLCVModel(
#             estimator=estimator,
#             evaluator=evaluator,
#             estimator_params=estimator_params,
#             evaluator_params=evaluator_params,
#             grid_params=grid_params,
#             cv_params=cv_params,
#         )
#
#         return ml_method
#
#     def get_eval_metric_name(self) -> str:
#         """
#         Get the evaluation metric name.
#
#         :return: The evaluation metric name.
#         """
#         return self.evaluator.getMetricName()
#
#     def get_feature_scores(self) -> pd.DataFrame:
#
#         # TODO: This function should be able to parse all available models.
#
#         indexed_features = self._fsdf.get_features_indexed()
#         best_model = self._get_best_model()
#
#         # raise exception if the model is not none
#         if best_model is None:
#             raise ValueError(
#                 "No ML model have been fitted. Use fit() to fit the model."
#             )
#
#         df_features = pd.DataFrame(indexed_features.to_numpy(), columns=["features"])
#
#         if isinstance(
#             best_model, (RandomForestClassificationModel, RandomForestRegressionModel)
#         ):
#             df_scores = pd.DataFrame(
#                 data=best_model.featureImportances.toArray(), columns=["scores"]
#             )
#
#             df_scores = df_scores.reset_index(level=0).rename(
#                 columns={"index": "feature_index"}
#             )
#
#             # merge the feature scores with the feature names
#             df = df_features.merge(
#                 df_scores, how="right", left_index=True, right_index=True
#             )  # index-to-index merging
#
#             # sort the dataframe by scores in descending order
#             df = df.sort_values(by="scores", ascending=False)
#
#             # add feature percentile rank to the features_scores dataframe
#             df["percentile_rank"] = df["scores"].rank(pct=True)
#
#             return df
#
#         else:
#             raise ValueError(
#                 "Unsupported model type. "
#                 "Only RandomForestClassificationModel, "
#                 "RandomForestRegressionModel, and LinearSVCModel are supported."
#             )
#
#     def get_eval_metric_on_training(self) -> float:
#         """
#         Get the evaluation metric on training data from a trained CrossValidatorModel (best model).
#
#         :return: A dictionary containing the evaluation metric name and value.
#         """
#
#         # TODO: This function should be able to parse all available models.
#
#         # get the best model from the fitted cross-validator model
#         best_model = self._get_best_model()
#
#         # get the eval metric name from the evaluator
#         eval_metric_name = self.get_eval_metric_name()
#
#         if isinstance(
#             best_model, (RandomForestClassificationModel, LogisticRegressionModel)
#         ):
#             metric_value = getattr(best_model.summary, eval_metric_name)
#
#         elif isinstance(best_model, LinearSVCModel):
#             metric_value = getattr(best_model.summary(), eval_metric_name)
#
#         else:
#             warnings.warn("Unsupported model type. Unable to get evaluation metric.")
#             metric_value = None
#
#         return metric_value
#
#     def get_eval_metric_on_testing(self, test_data: FSDataFrame) -> float:
#         """
#         Get accuracy on test data from a trained CrossValidatorModel (best model).
#
#         :param test_data: The test data as a FSDataFrame object.
#         :return: accuracy
#         """
#
#         # TODO: This function should be able to parse all available models.
#
#         # get the best model from the fitted cross-validator model
#         best_model = self._get_best_model()
#
#         # get test data features harmonized with training features
#         training_features = self._fsdf.get_features_names()
#         test_data = test_data.filter_features(training_features, keep=True)
#
#         # predict the test data
#         predictions = None
#         if isinstance(
#             best_model,
#             (RandomForestClassificationModel, LinearSVCModel, LogisticRegressionModel),
#         ):
#             predictions = best_model.transform(test_data.get_sdf_vector())
#
#         metric_value = None
#         if predictions is not None:
#             metric_value = self.evaluator.evaluate(predictions)
#
#         return metric_value
