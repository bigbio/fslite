"""
Example of a feature selection pipeline implemented in fslite.

After data import and pre-processing, the pipeline applies univariate correlation filter,
multivariate correlation filter and Randon Forest classification.

"""

from fslite.config.context import init_spark, stop_spark_session
from fslite.fs.core import FSDataFrame

from fslite.fs.methods import FSPipeline, FSUnivariate, FSMultivariate, FSMLMethod
from fslite.utils.datasets import get_tnbc_data_path
from fslite.utils.io import import_table_as_psdf

# Init spark
init_spark(
    apply_pyarrow_settings=True,
    apply_extra_spark_settings=True,
    apply_pandas_settings=True,
)

# 1. Import data
df = import_table_as_psdf(get_tnbc_data_path(), n_partitions=5)
fsdf = FSDataFrame(df, sample_col="Sample", label_col="label")

# 2. Split data
training_data, testing_data = fsdf.split_df(split_training_factor=0.6)

# 3. Set feature selection methods
# create a Univariate object
univariate = FSUnivariate(
    fs_method="anova", selection_mode="percentile", selection_threshold=0.8
)

# create a Multivariate object
multivariate = FSMultivariate(
    fs_method="m_corr", corr_threshold=0.75, corr_method="pearson"
)

# create a MLMethod object
rf_classifier = FSMLMethod(
    fs_method="rf_multilabel",
    rfe=True,
    rfe_iterations=2,
    percent_to_keep=0.9,
    estimator_params={"labelCol": "label"},
    evaluator_params={"metricName": "accuracy"},
    grid_params={"numTrees": [10, 15], "maxDepth": [5, 10]},
    cv_params={"parallelism": 2, "numFolds": 5},
)

# 4. Create a pipeline object
fs_pipeline = FSPipeline(
    df_training=training_data,
    df_testing=testing_data,
    fs_stages=[univariate, multivariate, rf_classifier],
)

# 5. Run the pipeline
results = fs_pipeline.run()

# Print results
print(f"Accuracy on training: {results['training_metric']}")
print(f"Accuracy on testing: {results['testing_metric']}")
print(results.get("feature_scores"))


stop_spark_session()
