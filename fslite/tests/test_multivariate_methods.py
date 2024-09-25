import pandas as pd

from fslite.fs.fdataframe import FSDataFrame
from fslite.fs.multivariate import FSMultivariate
from fslite.utils.datasets import get_tnbc_data_path


# test multivariate_filter method with 'm_corr' method
def test_multivariate_filter_corr_strict_mode():
    """
    Test multivariate_filter method with 'm_corr' method.
    :return: None
    """

    # import tsv as pandas DataFrame
    df = pd.read_csv(get_tnbc_data_path(), sep="\t")

    # create FSDataFrame instance
    fs_df = FSDataFrame(df=df, sample_col="Sample", label_col="label")

    # create FSMultivariate instance
    fs_multivariate = FSMultivariate(fs_method="m_corr",
                                     selection_mode="strict",
                                     selection_threshold=0.75)

    fsdf_filtered = fs_multivariate.select_features(fs_df)

    assert fs_df.count_features() == 500
    assert fsdf_filtered.count_features() == 239

    # Export the filtered DataFrame as Pandas DataFrame
    df_filtered = fsdf_filtered.to_pandas()
    df_filtered.to_csv("filtered_tnbc_data.csv", index=False)


# test multivariate_filter method with 'm_corr' method in approximate mode
def test_multivariate_filter_corr_approximate_mode():
    """
    Test multivariate_filter method with 'm_corr' method in approximate mode.
    :return: None
    """

    # import tsv as pandas DataFrame
    df = pd.read_csv(get_tnbc_data_path(), sep="\t")

    # create FSDataFrame instance
    fs_df = FSDataFrame(df=df, sample_col="Sample", label_col="label")

    # create FSMultivariate instance
    fs_multivariate = FSMultivariate(fs_method="m_corr",
                                     selection_mode="approximate",
                                     selection_threshold=0.75)

    fsdf_filtered = fs_multivariate.select_features(fs_df)

    assert fs_df.count_features() == 500

    # test if number of features selected is within the expected range [240-260]
    assert 240 <= fsdf_filtered.count_features() <= 260

    # Export the filtered DataFrame as Pandas DataFrame
    df_filtered = fsdf_filtered.to_pandas()
    df_filtered.to_csv("filtered_tnbc_data.csv", index=False)


# test multivariate_filter method with 'variance' method
def test_multivariate_filter_variance_percentile_mode():
    """
    Test multivariate_filter method with 'variance' method.
    :return: None
    """

    # import tsv as pandas DataFrame
    df = pd.read_csv(get_tnbc_data_path(), sep="\t")

    # create FSDataFrame instance
    fs_df = FSDataFrame(df=df, sample_col="Sample", label_col="label")

    # create FSMultivariate instance
    fs_multivariate = FSMultivariate(fs_method="variance",
                                     selection_mode="percentile",
                                     selection_threshold=0.2)

    fsdf_filtered = fs_multivariate.select_features(fs_df)

    assert fs_df.count_features() == 500
    assert fsdf_filtered.count_features() == 500

    # Export the filtered DataFrame as Pandas DataFrame
    df_filtered = fsdf_filtered.to_pandas()
    df_filtered.to_csv("filtered_tnbc_data.csv", index=False)


# test multivariate_filter method with 'variance' method in k_best mode
def test_multivariate_filter_variance_k_best_mode():
    """
    Test multivariate_filter method with 'variance' method in k_best mode.
    :return: None
    """

    # import tsv as pandas DataFrame
    df = pd.read_csv(get_tnbc_data_path(), sep="\t")

    # create FSDataFrame instance
    fs_df = FSDataFrame(df=df, sample_col="Sample", label_col="label")

    # create FSMultivariate instance
    fs_multivariate = FSMultivariate(fs_method="variance",
                                     selection_mode="k_best",
                                     selection_threshold=68100000.0
                                     # TODO: check this value (should be normalized variance?)
                                     )

    fsdf_filtered = fs_multivariate.select_features(fs_df)

    assert fs_df.count_features() == 500
    assert fsdf_filtered.count_features() == 87

    # Export the filtered DataFrame as Pandas DataFrame
    df_filtered = fsdf_filtered.to_pandas()
    df_filtered.to_csv("filtered_tnbc_data.csv", index=False)

