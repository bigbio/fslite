import logging
from typing import Dict, List

import numpy as np
from sklearn.feature_selection import (GenericUnivariateSelect,
                                       f_classif,
                                       f_regression,
                                       mutual_info_classif,
                                       mutual_info_regression)

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

    def univariate_feature_selector(
            self,
            df: FSDataFrame,
            score_func: str = "f_classif",
            selection_mode: str = "percentile",
            selection_threshold: float = 0.8
    ) -> List[int]:
        """
        Wrapper for scikit-learn's `GenericUnivariateSelect` feature selector, supporting multiple scoring functions.

        :param df: Input FSDataFrame
        :param score_func: The score function to use for feature selection. Options are:
                           - 'f_classif': ANOVA F-value for classification tasks.
                           - 'f_regression': F-value for regression tasks.
                           - 'mutual_info_classif': Mutual information for classification.
                           - 'mutual_info_regression': Mutual information for regression.
        :param selection_mode: Mode for feature selection (e.g. 'percentile' or 'k_best').
        :param selection_threshold: The percentage or number of features to select based on the selection mode.

        :return: List of selected feature indices.
        """
        # Define the score function based on input
        score_func_mapping = {
            "f_classif": f_classif,
            "f_regression": f_regression,
            "mutual_info_classif": mutual_info_classif,
            "mutual_info_regression": mutual_info_regression,
        }

        if score_func not in score_func_mapping:
            raise ValueError(f"Invalid score_func '{score_func}'. Valid options are: {list(score_func_mapping.keys())}")

        # Extract the score function
        selected_score_func = score_func_mapping[score_func]

        # Get the feature matrix and label vector from the FSDataFrame
        f_matrix = df.get_feature_matrix()
        y = df.get_label_vector()

        # Configure the selector using the provided score function and selection mode/threshold
        selector = GenericUnivariateSelect(score_func=selected_score_func,
                                           mode=selection_mode,
                                           param=selection_threshold)

        # Fit the selector and get only the selected feature indices (not the transformed matrix)
        _ = selector.fit_transform(f_matrix, y)
        selected_features = selector.get_support(indices=True)

        return list(selected_features)

    def univariate_filter(
            self, df: FSDataFrame, univariate_method: str = "u_corr", **kwargs
    ) -> FSDataFrame:
        """
        Filter features after applying a univariate feature selector method.

        :param df: Input DataFrame
        :param univariate_method: Univariate selector method ('u_corr', 'anova', 'f_regression',
                                                              'mutual_info_classification', 'mutual_info_regression')
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
            selected_features = self.univariate_feature_selector(df, score_func="f_classif", **kwargs)
        elif univariate_method == "f_regression":
            selected_features = self.univariate_feature_selector(df, score_func="f_regression", **kwargs)
        elif univariate_method == "u_corr":
            selected_features = univariate_correlation_selector(df, **kwargs)
        elif univariate_method == "mutual_info_classification":
            selected_features = self.univariate_feature_selector(df, score_func="mutual_info_classif", **kwargs)
        elif univariate_method == "mutual_info_regression":
            selected_features = self.univariate_feature_selector(df, score_func="mutual_info_regression", **kwargs)

        logger.info(
                    f"Applying univariate filter using method: {univariate_method} \n"
                    f" with selection mode: {kwargs.get('selection_mode')} \n"
                    f" and selection threshold: {kwargs.get('selection_threshold')}"
                    )

        if len(selected_features) == 0:
            logger.warning("No features selected. Returning original DataFrame.")
            return df
        else:
            logger.info(f"Selected {len(selected_features)} features...")
            return df.select_features_by_index(selected_features)


def univariate_correlation_selector(
    df: FSDataFrame,
    selection_threshold: float = 0.3
) -> List[int]:
    """
    TODO: Replace this implementation with sci-learn's GenericUnivariateSelect with score_func='f_regression'
    Select features based on their correlation with a label (class), if the correlation value is less than the specified
    threshold.

    :param df: Input DataFrame
    :param selection_threshold: Maximum allowed correlation threshold

    :return: List of selected feature indices
    """

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

    correlations = compute_univariate_corr(df)

    selected_features = [
        feature_index
        for feature_index, corr in correlations.items()
        if corr <= selection_threshold
    ]
    return selected_features

