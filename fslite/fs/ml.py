"""

A set of pre-defined ML algorithms wrapped with cross-validation approach
for feature selection (e.g., rank by feature importance) and prediction.

"""
from typing import Union, Optional, Dict, Any, List

from fslite.fs.constants import get_fs_ml_methods, is_valid_ml_method
from fslite.fs.fdataframe import FSDataFrame
from fslite.fs.methods import FSMethod, InvalidMethodError
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer
import pandas as pd

from fslite.fs.ml import MLCVModel


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

class MLCVModel:
    """
    A factory class for creating various machine learning models with scikit-learn.
    ML models are created using a cross-validator approach for hyperparameter tuning.
    """

    _grid_search: GridSearchCV = None
    _best_model: object = None
    _fsdf: FSDataFrame = None

    def __init__(
        self,
        estimator: Union[
            RandomForestClassifier,
            RandomForestRegressor,
            LinearSVC,
            LogisticRegression,
            SVC
        ],
        scoring: str,
        estimator_params: Optional[Dict[str, Any]] = None,
        grid_params: Optional[Dict[str, List[Any]]] = None,
        cv: int = 5,
    ):
        """
        Initializes the MLModel with optional estimator, scoring method, and parameter specifications.
        """
        self.estimator = estimator
        self.scoring = scoring
        self.estimator_params = estimator_params
        self.grid_params = grid_params
        self.cv = cv

        self._initialize_model()

    def _initialize_model(self):
        if self.estimator_params:
            self.estimator.set_params(**self.estimator_params)

        if self.grid_params:
            self._grid_search = GridSearchCV(
                estimator=self.estimator,
                param_grid=self.grid_params,
                scoring=self.scoring,
                cv=self.cv
            )

    def fit(self, fsdf: FSDataFrame) -> "MLCVModel":
        """
        Fit the model using the cross-validator.
        """
        self._fsdf = fsdf
        X, y = self._fsdf.get_features_and_labels()

        if self._grid_search:
            self._grid_search.fit(X, y)
            self._best_model = self._grid_search.best_estimator_
        else:
            self.estimator.fit(X, y)
            self._best_model = self.estimator

        return self

    def _get_best_model(self):
        if self._best_model is None:
            raise ValueError("No model has been fitted. Use fit() to fit the model.")
        return self._best_model

    def get_feature_scores(self) -> pd.DataFrame:
        """
        Get feature importance scores from the best model.
        """
        if not isinstance(self._best_model, (RandomForestClassifier, RandomForestRegressor)):
            raise ValueError("Feature importance is only available for tree-based models.")

        features = self._fsdf.get_feature_names()
        importances = self._best_model.feature_importances_
        df = pd.DataFrame({
            'feature': features,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

        return df

    def get_eval_metric_on_training(self) -> float:
        """
        Get the evaluation metric on training data from the best model.
        """
        X_train, y_train = self._fsdf.get_features_and_labels()
        y_pred = self._best_model.predict(X_train)

        if self.scoring == 'accuracy':
            return accuracy_score(y_train, y_pred)
        elif self.scoring == 'f1':
            return f1_score(y_train, y_pred)
        elif self.scoring == 'roc_auc':
            return roc_auc_score(y_train, y_pred)
        else:
            raise ValueError("Unsupported scoring method.")

    def get_eval_metric_on_testing(self, test_data: FSDataFrame) -> float:
        """
        Get evaluation metric on test data from the trained model.
        """
        X_test, y_test = test_data.get_features_and_labels()
        y_pred = self._best_model.predict(X_test)

        if self.scoring == 'accuracy':
            return accuracy_score(y_test, y_pred)
        elif self.scoring == 'f1':
            return f1_score(y_test, y_pred)
        elif self.scoring == 'roc_auc':
            return roc_auc_score(y_test, y_pred)
        else:
            raise ValueError("Unsupported scoring method.")

    @staticmethod
    def create_model(
        model_type: str,
        estimator_params: Dict[str, Any] = None,
        grid_params: Dict[str, List[Any]] = None,
        scoring: str = 'accuracy',
        cv: int = 5
    ) -> "MLCVModel":
        """
        Create an ML model based on the model type.
        """
        if model_type == "RF_BINARY":
            estimator = RandomForestClassifier()
        elif model_type == "LSVC_BINARY":
            estimator = LinearSVC()
        elif model_type == "RF_REGRESSION":
            estimator = RandomForestRegressor()
        elif model_type == "LOGISTIC_REGRESSION":
            estimator = LogisticRegression()
        else:
            raise ValueError(f"Unsupported model type: {model_type}.")

        return MLCVModel(
            estimator=estimator,
            scoring=scoring,
            estimator_params=estimator_params,
            grid_params=grid_params,
            cv=cv,
        )
