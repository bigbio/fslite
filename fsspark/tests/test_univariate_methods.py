import unittest

import pandas as pd
from fsspark.utils.datasets import get_tnbc_data_path
from fsspark.fs.fdataframe import FSDataFrame

from fsspark.fs.univariate import univariate_filter


class UnivariateMethodsTest(unittest.TestCase):
    """
    Define testing methods for FSDataFrame class.
    """

    def setUp(self) -> None:
        # import tsv as pandas DataFrame
        self.df = pd.read_csv(get_tnbc_data_path(), sep='\t')

        # create FSDataFrame instance
        self.fsdf = FSDataFrame(df=self.df,
                                sample_col='Sample',
                                label_col='label')

    def tearDown(self) -> None:
        pass

    def test_univariate_filter_corr(self):
        """
        Test univariate_filter method with 'u_corr' method.
        :return: None
        """

        fsdf = self.fsdf
        fsdf_filtered = univariate_filter(fsdf,
                                          univariate_method='u_corr',
                                          corr_threshold=0.3)

        self.assertEqual(fsdf.count_features(), 500)
        self.assertEqual(fsdf_filtered.count_features(), 211)

        # Export the filtered DataFrame as Pandas DataFrame
        df_filtered = fsdf_filtered.to_pandas()
        df_filtered.to_csv('filtered_tnbc_data.csv', index=False)

