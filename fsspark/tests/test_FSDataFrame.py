import unittest

from fsspark.config.context import init_spark, stop_spark_session
from fsspark.fs.core import FSDataFrame
from fsspark.utils.datasets import get_tnbc_data_path
from fsspark.utils.io import import_table_as_psdf


class FSDataFrameTest(unittest.TestCase):
    """
    Define testing methods for FSDataFrame class.
    """

    def setUp(self) -> None:
        init_spark(apply_pyarrow_settings=True,
                   apply_extra_spark_settings=True,
                   apply_pandas_settings=True)

    def tearDown(self) -> None:
        stop_spark_session()

    @staticmethod
    def import_FSDataFrame():
        df = import_table_as_psdf(get_tnbc_data_path(), n_partitions=5)
        fsdf = FSDataFrame(df, sample_col='Sample', label_col='label')
        return fsdf

    def test_FSDataFrame(self):
        """
        Test FSDataFrame class.
        :return: None
        """

        # create object of type FSDataFrame
        fsdf = self.import_FSDataFrame()

        self.assertEqual(len(fsdf.get_features_names()), 500)
        self.assertEqual(len(fsdf.get_sample_label()), 44)

    def test_get_sdf_vector(self):
        """
        Test get_sdf_vector method.
        :return: None
        """

        fsdf = self.import_FSDataFrame()

        sdf = fsdf.get_sdf_vector(output_column_vector='features')
        sdf.show(5)
        self.assertEqual(len(sdf.columns), 4)

    def test_scale_features(self):
        """
        Test scale_features method.
        :return: None
        """

        fsdf = self.import_FSDataFrame()
        fsdf = fsdf.scale_features(scaler_method='min_max')

        fsdf.get_sdf().show(10)
        self.assertGreaterEqual(min(fsdf.to_psdf()['tr|E9PBJ4'].to_numpy()), 0.0)
        self.assertLessEqual(max(fsdf.to_psdf()['tr|E9PBJ4'].to_numpy()), 1.0)

    def test_get_features_indices(self):
        """
        Test get_features_indices method.
        :return: None
        """

        fsdf = self.import_FSDataFrame()
        feature_indices = fsdf.get_features_indexed()
        feature_names = feature_indices.loc[[0, 1, 2, 5]].tolist()

        self.assertTrue(all([x in ['tr|E9PBJ4', 'sp|P07437', 'sp|P68371', 'tr|F8VWX4'] for x in feature_names]))


if __name__ == '__main__':
    unittest.main()
