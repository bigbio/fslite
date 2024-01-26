import unittest

import numpy as np

from fsspark.config.context import init_spark, stop_spark_session
from fsspark.fs.core import FSDataFrame
from fsspark.fs.utils import compute_missingness_rate, remove_features_by_missingness_rate, impute_missing
from fsspark.utils.datasets import get_tnbc_data_missing_values_path
from fsspark.utils.io import import_table_as_psdf


class TestDataPreprocessing(unittest.TestCase):
    """
    Define testing methods for data preprocessing (e.g, scaling, imputation, etc.)

    """

    def setUp(self) -> None:
        init_spark(apply_pyarrow_settings=True,
                   apply_extra_spark_settings=True,
                   apply_pandas_settings=True)

    def tearDown(self) -> None:
        stop_spark_session()

    @staticmethod
    def import_FSDataFrame() -> FSDataFrame:
        """
        Import FSDataFrame object with missing values.
        Number of samples: 44
        Number of features: 10 (5 with missing values)
        :return:
        """
        df = import_table_as_psdf(get_tnbc_data_missing_values_path(), n_partitions=5)
        fsdf = FSDataFrame(df, sample_col='Sample', label_col='label')
        return fsdf

    def test_compute_missingness_rate(self):
        """
        Test compute_missingness_rate method.
        :return: None
        """

        fsdf = self.import_FSDataFrame()
        features_missing_rates = compute_missingness_rate(fsdf)
        self.assertEqual(features_missing_rates.get('tr|E9PBJ4'), 0.0)
        self.assertAlmostEqual(features_missing_rates.get('sp|P07437'), 0.295, places=2)

    def test_filter_by_missingness_rate(self):
        """
        Test filter_missingness_rate method.
        :return: None
        """

        fsdf = self.import_FSDataFrame()
        fsdf = remove_features_by_missingness_rate(fsdf, threshold=0.15)
        # print number of features
        print(f"Number of remaining features: {fsdf.count_features()}")

        self.assertEqual(fsdf.count_features(), 6)

    def test_impute_missing(self):
        """
        Test impute_missing method. Impute missing values using the mean across columns.
        :return: None
        """

        fsdf = self.import_FSDataFrame()
        fsdf = impute_missing(fsdf, strategy='mean')

        # Collect features as array
        array = fsdf._collect_features_as_array()

        # Check if there are no missing (NaNs) or null values
        self.assertFalse(np.isnan(array).any())


if __name__ == '__main__':
    unittest.main()
