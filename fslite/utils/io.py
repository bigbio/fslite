import warnings

import pyspark.pandas
import pyspark.sql

from fslite.config.context import PANDAS_ON_SPARK_API_SETTINGS

warnings.filterwarnings("ignore")


def import_table(
    path: str, header: bool = True, sep: str = "\t", n_partitions: int = 5
) -> pyspark.sql.DataFrame:
    """
    Import tsv file as Spark DataFrame.

    :param path: File path
    :param header: True if the first row is header.
    :param sep: Column separator
    :param n_partitions: Minimal number of partitions

    :return: Spark DataFrame
    """

    _sc = pyspark.sql.SparkSession.getActiveSession()

    if _sc is None:
        raise ValueError("Active Spark Session not found...")

    sdf = (
        _sc.read.option("delimiter", sep)
        .option("header", header)
        .option("inferSchema", "true")
        .csv(path)
        .repartition(n_partitions)
    )
    return sdf


def import_parquet(path: str, header: bool = True) -> pyspark.sql.DataFrame:
    """
    Import parquet file as Spark DataFrame.

    :param path: File path
    :param header: True if the first row is header.

    :return: Spark DataFrame
    """

    _sc = pyspark.sql.SparkSession.getActiveSession()

    if _sc is None:
        raise ValueError("Active Spark Session not found...")

    sdf = _sc.read.option("header", header).option("inferSchema", "true").parquet(path)
    return sdf


def import_table_as_psdf(
    path: str, sep: str = "\t", n_partitions: int = 5
) -> pyspark.pandas.DataFrame:
    """
    Import tsv file as Pandas on Spark DataFrame

    :param path: Path to TSV file
    :param sep: Column separator (default: "\t")
    :param n_partitions: Minimal number of partitions

    :return: Pandas on Spark DataFrame
    """

    import pyspark.pandas as ps

    # apply settings for pandas on spark api
    [
        ps.set_option(k, PANDAS_ON_SPARK_API_SETTINGS.get(k))
        for k in PANDAS_ON_SPARK_API_SETTINGS.keys()
    ]

    psdf = ps.read_csv(path, sep=sep).spark.repartition(n_partitions)
    return psdf
