import unittest

import pyspark
import pyspark.pandas as ps

from fslite.config.context import init_spark, stop_spark_session
from fslite.utils.datasets import get_tnbc_data_path
from fslite.utils.io import import_table, import_table_as_psdf


class TestImportExport(unittest.TestCase):

    def setUp(self) -> None:
        init_spark(apply_pyarrow_settings=True,
                   apply_extra_spark_settings=True,
                   apply_pandas_settings=True)

    def tearDown(self) -> None:
        stop_spark_session()

    def test_import_tsv(self):
        """
        Test import tsv file as Spark DataFrame.
        :return: None
        """
        df = import_table(path=get_tnbc_data_path(),
                          n_partitions=10)

        self.assertIsInstance(df, pyspark.sql.DataFrame)
        self.assertEqual(df.count(), 44)

    def test_import_tsv_as_psdf(self):
        """
        Test import tsv file as Pandas on Spark DataFrame (PoS).
        :return: None
        """
        df = import_table_as_psdf(path=get_tnbc_data_path(),
                                  n_partitions=10)

        self.assertIsInstance(df, ps.frame.DataFrame)
        self.assertEqual(df.shape,  (44, 502))


if __name__ == '__main__':
    unittest.main()
