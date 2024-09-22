import logging
from typing import List

import numpy as np
import pyspark
from pyspark.ml.feature import VarianceThresholdSelector
from pyspark.ml.stat import Correlation

from fslite.fs.constants import (
    MULTIVARIATE_METHODS,
    MULTIVARIATE_CORRELATION,
    MULTIVARIATE_VARIANCE,
)

from fslite.fs.core import FSDataFrame
from fslite.fs.utils import find_maximal_independent_set
from fslite.utils.generic import tag

logging.basicConfig(format="%(levelname)s (%(name)s %(lineno)s): %(message)s")
logger = logging.getLogger("FSSPARK:MULTIVARIATE")
logger.setLevel(logging.INFO)


@tag("experimental")
def _compute_correlation_matrix(
    sdf: pyspark.sql.DataFrame,
    features_col: str = "features",
    corr_method: str = "pearson",
) -> np.ndarray:
    """
    Compute features Matrix Correlation.

    :param sdf: Spark DataFrame
    :param features_col: Name of the feature column vector name.
    :param corr_method: One of `pearson` (default) or `spearman`.

    :return: Numpy array.
    """

    logger.warning(
        "Warning: Computed matrix correlation will be collected into the drive with this implementation.\n"
        "This may cause memory issues. Use it preferably with small datasets."
    )
    logger.info(f"Computing correlation matrix using {corr_method} method.")

    mcorr = Correlation.corr(sdf, features_col, corr_method).collect()[0][0].toArray()
    return mcorr


@tag("experimental")
def multivariate_correlation_selector(
    fsdf: FSDataFrame,
    strict: bool = True,
    corr_threshold: float = 0.75,
    corr_method: str = "pearson",
) -> List[str]:
    """
    Compute the correlation matrix (Pearson) among input features and select those below a specified threshold.

    :param fsdf: Input FSDataFrame
    :param strict: If True (default), apply hard filtering (strict) to remove highly correlated features.
                   Otherwise, find the maximal independent set of highly correlated features (approximate method).
                   `Warning`: The approximate method is experimental.
    :param corr_threshold: Minimal correlation threshold to consider two features correlated.
    :param corr_method: One of `pearson` (default) or `spearman`.

    :return: List of selected features names
    """

    colum_vector_features = "features"
    sdf = fsdf.get_sdf_vector(output_column_vector=colum_vector_features)

    # compute correlation matrix
    mcorr = _compute_correlation_matrix(
        sdf, features_col=colum_vector_features, corr_method=corr_method
    )

    mcorr = np.abs(mcorr)  # get absolute correlation value
    combs_above_cutoff = (
        np.triu(mcorr, k=1) > corr_threshold
    )  # create bool matrix that meet criteria
    correlated_col_index = tuple(
        np.column_stack(np.where(combs_above_cutoff))
    )  # get correlated pairs cols index

    index_to_remove = set()
    if strict:
        # hard filtering method
        # Original implementation: https://www.rdocumentation.org/packages/caret/versions/6.0-93/topics/findCorrelation
        cols_mean = np.mean(mcorr, axis=1)  # get cols index mean
        for pairs in correlated_col_index:
            i = pairs[0]
            j = pairs[1]
            index_to_remove.add(i if cols_mean[i] > cols_mean[j] else j)
    else:
        # approximate method
        index_to_remove = find_maximal_independent_set(correlated_col_index, keep=False)

    features = fsdf.get_features_names()  # get all current features
    features_to_remove = fsdf.get_features_by_index(index_to_remove)
    selected_features = [sf for sf in features if sf not in features_to_remove]

    return selected_features


@tag("spark implementation")
def multivariate_variance_selector(
    fsdf: FSDataFrame, variance_threshold: float = 0.0
) -> List[str]:
    """
    Select features after removing low-variance ones (e.g., features with quasi-constant value across samples).

    :param fsdf: Input FSDataFrame
    :param variance_threshold: Minimal variance value allowed to select a feature.

    :return: List of selected features names
    """

    colum_vector_features = "features"
    sdf = fsdf.get_sdf_vector(output_column_vector=colum_vector_features)

    selector = VarianceThresholdSelector()
    (
        selector.setFeaturesCol(colum_vector_features)
        .setOutputCol("selectedFeatures")
        .setVarianceThreshold(variance_threshold)
    )

    model = selector.fit(sdf)
    selected_features_indices = set(model.selectedFeatures)
    selected_features = fsdf.get_features_by_index(selected_features_indices)

    return selected_features


def multivariate_filter(
    fsdf: FSDataFrame, multivariate_method: str = "m_corr", **kwargs
) -> FSDataFrame:
    """
     Filter features after applying a multivariate feature selector method.

    :param fsdf: Input FSDataFrame
    :param multivariate_method: Multivariate selector method.
                                Possible values are 'm_corr' or 'variance'.

    :return: Filtered FSDataFrame
    """
    if multivariate_method == MULTIVARIATE_CORRELATION:
        selected_features = multivariate_correlation_selector(fsdf, **kwargs)
    elif multivariate_method == MULTIVARIATE_VARIANCE:
        selected_features = multivariate_variance_selector(fsdf, **kwargs)
    else:
        raise ValueError(
            f"Invalid multivariate method: {multivariate_method}. "
            f"Choose one of {MULTIVARIATE_METHODS.keys()}."
        )

    logger.info(f"Applying multivariate filter {multivariate_method}.")

    return fsdf.filter_features(selected_features, keep=True)
