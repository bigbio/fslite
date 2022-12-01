"""
Example of a feature selection pipeline implemented in fsspark.

After data import and pre-processing, the pipeline applies univariate correlation filter,
multivariate correlation filter and Randon Forest classification.

"""

from fsspark.context import init_spark, stop_spark_session
from fsspark.fs.core import FSDataFrame
from fsspark.fs.ml import cv_rf_classification, get_accuracy, get_predictions
from fsspark.fs.multivariate import multivariate_filter
from fsspark.fs.univariate import univariate_filter
from fsspark.fs.utils import filter_missingness_rate, impute_missing
from fsspark.utils.datasets import get_tnbc_data_path
from fsspark.utils.io import import_table_as_psdf

# Init spark
init_spark()

# Import data
fsdf = import_table_as_psdf(get_tnbc_data_path(), n_partitions=5)

fsdf = FSDataFrame(fsdf, sample_col="Sample", label_col="label")

# Step 1. Data pre-processing.

# a) Filter missingness rate
fsdf = filter_missingness_rate(fsdf, threshold=0.1)

# b) Impute data frame
fsdf = impute_missing(fsdf)

# c) Scale features
fsdf = fsdf.scale_features(scaler_method="standard")

# Split dataset in training/testing
training_df, testing_df = fsdf.split_df(label_type_cat=True, split_training_factor=0.8)

# Step 2. Apply univariate correlation filter
training_df = univariate_filter(
    training_df, univariate_method="u_corr", corr_threshold=0.3
)

# Step 3. Apply multivariate correlation filter
training_df = multivariate_filter(
    training_df, multivariate_method="m_corr", corr_threshold=0.7
)

# Step 4. ML-algorithm with cross-validation
cv_model = cv_rf_classification(training_df, binary_classification=False)

# Print out some stats

# Get accuracy from training
acc = get_accuracy(model=cv_model)
print(f"Training accuracy: {acc}")

# Get predictions from training
pred = get_predictions(model=cv_model)
pred.show()

stop_spark_session()
