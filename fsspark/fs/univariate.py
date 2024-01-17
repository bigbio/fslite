import logging
from typing import Dict, List

import pyspark.sql.functions as f
from pyspark.ml.feature import UnivariateFeatureSelector

from fsspark.fs.core import FSDataFrame
from fsspark.utils.generic import tag

logging.basicConfig(format="%(levelname)s (%(name)s %(lineno)s): %(message)s")
logger = logging.getLogger("FSSPARK:UNIVARIATE")
logger.setLevel(logging.INFO)


@tag("spark implementation")
def compute_univariate_corr(fsdf: FSDataFrame) -> Dict[str, float]:
    """
    Compute the correlation coefficient between every column (features) in the input DataFrame and the label (class).

    :param fsdf: Input FSDataFrame

    :return: Return dict {feature -> corr}
    """

    sdf = fsdf.get_sdf()
    features = fsdf.get_features_names()
    label = fsdf.get_label_col_name()

    u_corr = sdf.select([f.abs(f.corr(sdf[c], sdf[label])).alias(c) for c in features])

    return u_corr.first().asDict()


def univariate_correlation_selector(fsdf: FSDataFrame,
                                    corr_threshold: float = 0.3) -> List[str]:
    """
    Select features based on its correlation with a label (class), if corr value is less than a specified threshold.
    Expected both features and label to be of type numeric.

    :param fsdf: FSDataFrame
    :param corr_threshold: Maximal correlation threshold allowed between feature and label.

    :return: List of selected features names
    """
    d = compute_univariate_corr(fsdf)
    selected_features = [k for k in d.keys() if d.get(k) <= corr_threshold]

    return selected_features


@tag("spark implementation")
def univariate_selector(fsdf: FSDataFrame,
                        label_type: str = 'categorical',
                        **kwargs) -> List[str]:
    """
    Wrapper for `UnivariateFeatureSelector`.
    See https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.UnivariateFeatureSelector.html

    Only continues features are supported. If label is categorical, ANOVA test is used. If label is of type continues
    then an F-regression test is used.

    :param fsdf: Input FSDataFrame
    :param label_type: Type of label. Possible values are 'categorical' or 'continuous'.

    :return: List of selected features names
    """

    vector_col_name = 'features'
    sdf = fsdf.get_sdf_vector(output_column_vector=vector_col_name)
    label = fsdf.get_label_col_name()

    # set selector
    if label_type == 'categorical':
        # TODO: print msg to logger with the method being used here...
        print("ANOVA (F-classification) univariate feature selection")
    elif label_type == 'continuous':
        # TODO: print msg to logger with the method being used here...
        print("F-value (F-regression) univariate feature selection")
    else:
        raise ValueError("`label_type` must be one of categorical or continuous")

    selector = UnivariateFeatureSelector(**kwargs)
    (selector
     .setLabelType(label_type)
     .setFeaturesCol(vector_col_name)
     .setFeatureType("continuous")
     .setOutputCol("selectedFeatures")
     .setLabelCol(label)
     )

    model = selector.fit(sdf)
    selected_features_indices = model.selectedFeatures
    selected_features = fsdf.get_features_by_index(selected_features_indices)

    return selected_features


@tag("spark implementation")
def univariate_filter(fsdf: FSDataFrame,
                      univariate_method: str = 'u_corr',
                      **kwargs) -> FSDataFrame:
    """
    Filter features after applying a univariate feature selector method.

    :param fsdf: Input FSDataFrame
    :param univariate_method: Univariate selector method.
                              Possible values are 'u_corr', 'anova' (categorical label)
                              or  'f_regression' (continuous label).

    :return: Filtered FSDataFrame
    """

    if univariate_method == 'anova':
        selected_features = univariate_selector(fsdf, label_type='categorical', **kwargs)
    elif univariate_method == 'f_regression':
        selected_features = univariate_selector(fsdf, label_type='continuous', **kwargs)
    elif univariate_method == 'u_corr':
        selected_features = univariate_correlation_selector(fsdf, **kwargs)
    else:
        raise ValueError("`method` must be one of anova, f_regression or u_corr.")

    logger.info(f"Applying univariate filter {univariate_method}.")

    return fsdf.filter_features(selected_features, keep=True)
