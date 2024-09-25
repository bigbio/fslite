import pandas as pd

from fslite.fs.fdataframe import FSDataFrame
from fslite.fs.univariate import FSUnivariate
from fslite.utils.datasets import get_tnbc_data_path


def test_univariate_filter_corr():
    """
    Test univariate_filter method with 'u_corr' method.
    :return: None
    """

    # import tsv as pandas DataFrame
    df = pd.read_csv(get_tnbc_data_path(), sep="\t")

    # create FSDataFrame instance
    fs_df = FSDataFrame(df=df, sample_col="Sample", label_col="label")

    # create FSUnivariate instance
    fs_univariate = FSUnivariate(fs_method="u_corr", selection_threshold=0.3)

    fsdf_filtered = fs_univariate.select_features(fs_df)

    assert fs_df.count_features() == 500
    assert fsdf_filtered.count_features() == 211

    # Export the filtered DataFrame as Pandas DataFrame
    df_filtered = fsdf_filtered.to_pandas()
    df_filtered.to_csv("filtered_tnbc_data.csv", index=False)


# test the univariate_filter method with 'anova' method
def test_univariate_filter_anova():
    """
    Test univariate_filter method with 'anova' method.
    :return: None
    """

    # import tsv as pandas DataFrame
    df = pd.read_csv(get_tnbc_data_path(), sep="\t")

    # create FSDataFrame instance
    fs_df = FSDataFrame(df=df, sample_col="Sample", label_col="label")

    # create FSUnivariate instance
    fs_univariate = FSUnivariate(
        fs_method="anova", selection_mode="percentile", selection_threshold=0.8
    )

    fsdf_filtered = fs_univariate.select_features(fs_df)

    assert fs_df.count_features() == 500
    assert fsdf_filtered.count_features() == 4

    # Export the filtered DataFrame as Pandas DataFrame
    df_filtered = fsdf_filtered.to_pandas()
    df_filtered.to_csv("filtered_tnbc_data.csv", index=False)


# test the univariate_filter method with 'mutual_info_classification' method
def test_univariate_filter_mutual_info_classification():
    """
    Test univariate_filter method with 'mutual_info_classification' method.
    :return: None
    """

    # import tsv as pandas DataFrame
    df = pd.read_csv(get_tnbc_data_path(), sep="\t")

    # create FSDataFrame instance
    fs_df = FSDataFrame(df=df, sample_col="Sample", label_col="label")

    # create FSUnivariate instance
    fs_univariate = FSUnivariate(
        fs_method="mutual_info_classification",
        selection_mode="k_best",
        selection_threshold=30,
    )

    fsdf_filtered = fs_univariate.select_features(fs_df)

    assert fs_df.count_features() == 500
    assert fsdf_filtered.count_features() == 30

    # Export the filtered DataFrame as Pandas DataFrame
    df_filtered = fsdf_filtered.to_pandas()
    df_filtered.to_csv("filtered_tnbc_data.csv", index=False)


# test the univariate_filter method with 'mutual_info_regression' method
def test_univariate_filter_mutual_info_regression():
    """
    Test univariate_filter method with 'mutual_info_regression' method.
    :return: None
    """

    # import tsv as pandas DataFrame
    df = pd.read_csv(get_tnbc_data_path(), sep="\t")

    # create FSDataFrame instance
    fs_df = FSDataFrame(df=df, sample_col="Sample", label_col="label")

    # create FSUnivariate instance
    fs_univariate = FSUnivariate(
        fs_method="mutual_info_regression",
        selection_mode="percentile",
        selection_threshold=0.8,
    )

    fsdf_filtered = fs_univariate.select_features(fs_df)

    assert fs_df.count_features() == 500
    assert fsdf_filtered.count_features() == 4

    # Export the filtered DataFrame as Pandas DataFrame
    df_filtered = fsdf_filtered.to_pandas()
    df_filtered.to_csv("filtered_tnbc_data.csv", index=False)


# test the univariate_filter method with f-regression method
def test_univariate_filter_f_regression():
    """
    Test univariate_filter method with f_regression method.
    :return: None
    """

    # import tsv as pandas DataFrame
    df = pd.read_csv(get_tnbc_data_path(), sep="\t")

    # create FSDataFrame instance
    fs_df = FSDataFrame(df=df, sample_col="Sample", label_col="label")

    # create FSUnivariate instance
    fs_univariate = FSUnivariate(
        fs_method="f_regression", selection_mode="percentile", selection_threshold=0.8
    )

    fsdf_filtered = fs_univariate.select_features(fs_df)

    assert fs_df.count_features() == 500
    assert fsdf_filtered.count_features() == 4

    # Export the filtered DataFrame as Pandas DataFrame
    df_filtered = fsdf_filtered.to_pandas()
    df_filtered.to_csv("filtered_tnbc_data.csv", index=False)
