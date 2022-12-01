import unittest

from fsspark.context import init_spark
from fsspark.fs.core import FSDataFrame
from fsspark.utils.datasets import get_tnbc_data_path
from fsspark.utils.io import import_table, import_table_as_psdf


class FSDataFrameTest(unittest.TestCase):
    """
    Define testing methods for FSDataFrame class.
    """

    def test_import_tsv(self):
        df = import_table(path=get_tnbc_data_path(),
                          n_partitions=20)

        self.assertEqual(df.count(), 44)

    @staticmethod
    def import_FSDataFrame():
        init_spark()
        df = import_table_as_psdf(get_tnbc_data_path(), n_partitions=5)
        fsdf = FSDataFrame(df, sample_col='Sample', label_col='label')
        return fsdf

    def test_FSDataFrame(self):
        # create object of type FSDataFrame
        fsdf = self.import_FSDataFrame()

        self.assertEqual(len(fsdf.get_features_names()), 500)
        self.assertEqual(len(fsdf.get_sample_label()), 44)

    def test_get_sdf_vector(self):
        fsdf = self.import_FSDataFrame()

        sdf = fsdf.get_sdf_vector(output_column_vector='features')
        sdf.show(5)
        self.assertEqual(len(sdf.columns), 4)

    def test_scale_features(self):
        fsdf = self.import_FSDataFrame()
        fsdf = fsdf.scale_features()

        fsdf.get_sdf().show(10)

    def test_get_features_indices(self):
        fsdf = self.import_FSDataFrame()
        f_indices = fsdf.get_features_indexed()
        l = f_indices.loc[[0, 1, 2, 5]].tolist()
        print(l)


if __name__ == '__main__':
    unittest.main()
