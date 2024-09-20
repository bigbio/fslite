import logging
from typing import Dict, List

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from scipy.stats import pearsonr

logging.basicConfig(format="%(levelname)s (%(name)s %(lineno)s): %(message)s")
logger = logging.getLogger("FS:UNIVARIATE")
logger.setLevel(logging.INFO)


def compute_univariate_corr(df: pd.DataFrame, features: List[str], label: str) -> Dict[str, float]:
    """
    Compute the correlation coefficient between every column (features) in the input DataFrame and the label (class).

    :param df: Input DataFrame
    :param features: List of feature column names
    :param label: Label column name

    :return: Return dict {feature -> corr}
    """
    correlations = {feature: abs(df[feature].corr(df[label])) for feature in features}
    return correlations


def univariate_correlation_selector(df: pd.DataFrame, features: List[str], label: str, corr_threshold: float = 0.3) -> \
List[str]:
    """
    Select features based on their correlation with a label (class), if the correlation value is less than the specified threshold.

    :param df: Input DataFrame
    :param features: List of feature column names
    :param label: Label column name
    :param corr_threshold: Maximum allowed correlation threshold

    :return: List of selected feature names
    """
    correlations = compute_univariate_corr(df, features, label)
    selected_features = [feature for feature, corr in correlations.items() if corr <= corr_threshold]
    return selected_features


def univariate_selector(df: pd.DataFrame, features: List[str], label: str, label_type: str = 'categorical',
                        selection_mode: str = 'percentile', selection_threshold: float = 0.8) -> List[str]:
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

    if label_type == 'categorical':
        logger.info("ANOVA (F-classification) univariate feature selection")
        selector = SelectKBest(score_func=f_classif)
    elif label_type == 'continuous':
        logger.info("F-value (F-regression) univariate feature selection")
        selector = SelectKBest(score_func=f_regression)
    else:
        raise ValueError("`label_type` must be one of 'categorical' or 'continuous'")

    if selection_mode == 'percentile':
        selector.set_params(k='all')  # We'll handle the percentile threshold manually
        selector.fit(X, y)
        scores = selector.scores_
        selected_indices = [i for i, score in enumerate(scores) if score >= selection_threshold * max(scores)]
    else:
        selector.set_params(k=int(selection_threshold))
        selector.fit(X, y)
        selected_indices = selector.get_support(indices=True)

    selected_features = [features[i] for i in selected_indices]
    return selected_features


def univariate_filter(df: pd.DataFrame, features: List[str], label: str, univariate_method: str = 'u_corr',
                      **kwargs) -> pd.DataFrame:
    """
    Filter features after applying a univariate feature selector method.

    :param df: Input DataFrame
    :param features: List of feature column names
    :param label: Label column name
    :param univariate_method: Univariate selector method ('u_corr', 'anova', 'f_regression')

    :return: Filtered DataFrame with selected features
    """

    if univariate_method == 'anova':
        selected_features = univariate_selector(df, features, label, label_type='categorical', **kwargs)
    elif univariate_method == 'f_regression':
        selected_features = univariate_selector(df, features, label, label_type='continuous', **kwargs)
    elif univariate_method == 'u_corr':
        selected_features = univariate_correlation_selector(df, features, label, **kwargs)
    else:
        raise ValueError(f"Univariate method {univariate_method} not supported.")

    logger.info(f"Applying univariate filter using method: {univariate_method}")

    return df[selected_features + [label]]  # Return DataFrame with selected features and label column
