import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

from fslite.fs.constants import get_fs_univariate_methods, is_valid_univariate_method
from fslite.fs.fdataframe import FSDataFrame
from fslite.fs.methods import FSMethod, InvalidMethodError

logging.basicConfig(format="%(levelname)s (%(name)s %(lineno)s): %(message)s")
logger = logging.getLogger("FS:UNIVARIATE")
logger.setLevel(logging.INFO)


class FSUnivariate(FSMethod):
    """
    A class for univariate feature selection methods.
    """

    valid_methods = get_fs_univariate_methods()

    def __init__(self, fs_method: str, **kwargs):
        """
        Initialize the univariate feature selection method with the specified parameters.

        :param fs_method: The univariate method to be used for feature selection.
        :param kwargs: Additional keyword arguments for the feature selection method.
        """
        super().__init__(fs_method, **kwargs)
        self.validate_method(fs_method)

    def validate_method(self, fs_method: str):
        """
        Validate the univariate method.

        :param fs_method: The univariate method to be validated.
        """

        if not is_valid_univariate_method(fs_method):
            raise InvalidMethodError(
                f"Invalid univariate method: {fs_method}. "
                f"Accepted methods are {', '.join(self.valid_methods)}"
            )

    def select_features(self, fsdf) -> FSDataFrame:
        """
        Select features using the specified univariate method.

        :param fsdf: The data frame on which feature selection is to be performed.
        :return fsdf: The data frame with selected features.
        """

        return self.univariate_filter(
            fsdf, univariate_method=self.fs_method, **self.kwargs
        )

    def __str__(self):
        return f"FSUnivariate(method={self.fs_method}, kwargs={self.kwargs})"

    def __repr__(self):
        return self.__str__()

    def univariate_filter(
        self, df: FSDataFrame, univariate_method: str = "u_corr", **kwargs
    ) -> FSDataFrame:
        """
        Filter features after applying a univariate feature selector method.

        :param df: Input DataFrame
        :param univariate_method: Univariate selector method ('u_corr', 'anova', 'f_regression')
        :return: Filtered DataFrame with selected features
        """

        if not is_valid_univariate_method(univariate_method):
            raise NotImplementedError(
                "The provided method {} is not implemented !! please select one from this list {}".format(
                    univariate_method, get_fs_univariate_methods()
                )
            )

        selected_features = []

        if univariate_method == "anova":
            # TODO: Implement ANOVA selector
            # selected_features = univariate_selector(df, features, label, label_type='categorical', **kwargs)
            pass
        elif univariate_method == "f_regression":
            # TODO: Implement F-regression selector
            # selected_features = univariate_selector(df, features, label, label_type='continuous', **kwargs)
            pass
        elif univariate_method == "u_corr":
            selected_features = univariate_correlation_selector(df, **kwargs)

        logger.info(f"Applying univariate filter using method: {univariate_method}")

        if len(selected_features) == 0:
            logger.warning("No features selected. Returning original DataFrame.")
            return df
        else:
            logger.info(f"Selected {len(selected_features)} features...")
            return df.select_features_by_index(selected_features)


def compute_univariate_corr(df: FSDataFrame) -> Dict[int, float]:
    """
    Compute the correlation coefficient between every column (features) in the input NumPy array and the label (class)
    using a dictionary comprehension.

    :param df: Input FSDataFrame
    :return: Return dict {feature_index -> corr}
    """

    f_matrix = df.get_feature_matrix()  # get the feature matrix
    labels = df.get_label_vector()  # get the label vector
    features_index = range(f_matrix.shape[1])  # get the feature index

    return {
        f_index: abs(np.corrcoef(f_matrix[:, f_index], labels)[0, 1])
        for f_index in features_index
    }


def univariate_correlation_selector(
    df: FSDataFrame, corr_threshold: float = 0.3
) -> List[int]:
    """
    Select features based on their correlation with a label (class), if the correlation value is less than the specified
    threshold.

    :param df: Input DataFrame
    :param corr_threshold: Maximum allowed correlation threshold

    :return: List of selected feature indices
    """
    correlations = compute_univariate_corr(df)
    selected_features = [
        feature_index
        for feature_index, corr in correlations.items()
        if corr <= corr_threshold
    ]
    return selected_features


def univariate_selector(
    df: pd.DataFrame,
    features: List[str],
    label: str,
    label_type: str = "categorical",
    selection_mode: str = "percentile",
    selection_threshold: float = 0.8,
) -> List[str]:
    """
    Wrapper for scikit-learn's `SelectKBest` feature selector.
    If the label is categorical, ANOVA test is used; if continuous, F-regression test is used.

    :param df: Input DataFrame
    :param features: List of feature column names
    :param label: Label column name
    :param label_type: Type of label ('categorical' or 'continuous')
    :param selection_mode: Mode for feature selection ('percentile' or 'k_best')
    :param selection_threshold: Number of features to select or the percentage of features

    :return: List of selected feature names
    """

    X = df[features].values
    y = df[label].values

    if label_type == "categorical":
        logger.info("ANOVA (F-classification) univariate feature selection")
        selector = SelectKBest(score_func=f_classif)
    elif label_type == "continuous":
        logger.info("F-value (F-regression) univariate feature selection")
        selector = SelectKBest(score_func=f_regression)
    else:
        raise ValueError("`label_type` must be one of 'categorical' or 'continuous'")

    if selection_mode == "percentile":
        selector.set_params(k="all")  # We'll handle the percentile threshold manually
        selector.fit(X, y)
        scores = selector.scores_
        selected_indices = [
            i
            for i, score in enumerate(scores)
            if score >= selection_threshold * max(scores)
        ]
    else:
        selector.set_params(k=int(selection_threshold))
        selector.fit(X, y)
        selected_indices = selector.get_support(indices=True)

    selected_features = [features[i] for i in selected_indices]
    return selected_features
