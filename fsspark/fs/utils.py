import logging
from typing import Dict, Tuple, Set

import networkx as nx
import pyspark.sql.functions as f
from networkx.algorithms.mis import maximal_independent_set
from pyspark.ml.feature import Imputer

from fsspark.fs.core import FSDataFrame
from fsspark.utils.generic import tag

logging.basicConfig(format="%(levelname)s (%(name)s %(lineno)s): %(message)s")
logger = logging.getLogger("FSSPARK:UTILS")
logger.setLevel(logging.INFO)


@tag("spark implementation")
def compute_missingness_rate(fsdf: FSDataFrame) -> Dict[str, float]:
    """
    Compute per features missingness rate.

    :param fsdf: FSDataFrame.
    :return: Dict with features (keys) and missingness rates (values).
    """

    sdf = fsdf.get_sdf()  # current Spark DataFrame.
    n_instances = fsdf.count_instances()  # number of instances/samples.
    features = fsdf.get_features_names()  # list of features (column) names

    missing_rates = sdf.select(
        [
            (
                    f.sum(f.when(f.isnan(sdf[c]) | f.isnull(sdf[c]), 1).otherwise(0)) / n_instances
            ).alias(c)
            for c in features
        ]
    )

    return missing_rates.first().asDict()


def remove_features_by_missingness_rate(
        fsdf: FSDataFrame, threshold: float = 0.15
) -> FSDataFrame:
    """
    Remove features from FSDataFrame with missingness rate higher or equal than a specified threshold.

    :param fsdf: FSDataFrame.
    :param threshold: maximal missingness rate allowed to keep a feature.
    :return: FSDataFrame with removed features.
    """
    d_rates = compute_missingness_rate(fsdf)
    features_to_remove = [k for k in d_rates.keys() if d_rates.get(k) >= threshold]

    logger.info(f"Applying missingness rate filter with threshold at {threshold}.")

    fsdf_filtered = fsdf.filter_features(features=features_to_remove, keep=False)

    return fsdf_filtered


@tag("spark implementation")
def impute_missing(fsdf: FSDataFrame, strategy: str = "mean") -> FSDataFrame:
    """
    Impute missing values using the mean, median or mode.
    Missing values are imputed column-wise (a.k.a features).

    :param fsdf: FSDataFrame
    :param strategy: Imputation method (mean, median or mode).

    :return: FSDataFrame with imputed values
    """

    if strategy not in ("mean", "median", "mode"):
        raise ValueError("Imputation method must be one of mean, median or mode...")

    sdf = fsdf.get_sdf()
    col_features = fsdf.get_features_names()

    sdf_imputed = (
        Imputer()
        .setStrategy(strategy)
        .setInputCols(col_features)
        .setOutputCols(col_features)
        .fit(sdf)
        .transform(sdf)
    )

    logger.info(f"Imputing features missing values with method {strategy}.")

    return FSDataFrame(
        sdf_imputed,
        fsdf.get_sample_col_name(),
        fsdf.get_label_col_name(),
        fsdf.get_row_index_name(),
    )


@tag("experimental")
def find_maximal_independent_set(pairs: Tuple[int], keep: bool = True) -> Set[int]:
    """
    Given a set of indices pairs, returns a random maximal independent set.

    :param pairs: Set of indices pairs.
    :param keep: If true (default), return the maximal independent set.
                 Otherwise, return the remaining indices after removing the maximal independent set.

    :return: Set of indices (maximal independent set or remaining indices).
    """
    logger.warning("This method is experimental and have been not extensively tested...")

    graph = nx.Graph()
    graph.add_edges_from(pairs)
    max_ind_set = maximal_independent_set(graph)
    if keep:
        return set(max_ind_set)
    else:
        return set([int(i) for i in graph.nodes if i not in max_ind_set])
