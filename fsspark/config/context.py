import os
import pyspark
from pyspark.sql import SparkSession

from fsspark.config.global_settings import (SPARK_EXTRA_SETTINGS,
                                            PYARROW_SETTINGS,
                                            PANDAS_ON_SPARK_API_SETTINGS)

os.environ['PYARROW_IGNORE_TIMEZONE'] = "1"


# os.environ['JAVA_HOME'] = "/Library/Java/JavaVirtualMachines/jdk1.8.0_162.jdk/Contents/Home"
# os.environ['SPARK_HOME'] = "/usr/local/spark-3.3.0-bin-hadoop3"

def init_spark(master: str = "local[8]",
               apply_pyarrow_settings: bool = True,
               apply_extra_spark_settings: bool = True,
               apply_pandas_settings: bool = True) -> SparkSession:
    """
    Init Spark session.

    :return: Spark session
    """
    # stop any current session before starting a new one.
    # stop_spark_session()

    # init or get spark session.
    spark = (SparkSession.builder
             .master(master)
             .appName("fs-spark")
             )

    if apply_extra_spark_settings:
        # Spark must be configured before starting context.
        for k in SPARK_EXTRA_SETTINGS.keys():
            spark = spark.config(k, SPARK_EXTRA_SETTINGS.get(k))
        spark = spark.getOrCreate()
    else:
        spark = spark.getOrCreate()

    if apply_pyarrow_settings:
        [spark.conf.set(k, PYARROW_SETTINGS.get(k)) for k in PYARROW_SETTINGS.keys()]
    if apply_pandas_settings:
        [spark.conf.set(k, PANDAS_ON_SPARK_API_SETTINGS.get(k)) for k in PANDAS_ON_SPARK_API_SETTINGS.keys()]

    return spark


def stop_spark_session() -> None:
    """
    If any, stop active Spark Session.

    :return: None
    """
    sc = pyspark.sql.SparkSession.getActiveSession()
    if sc is not None:
        sc.stop()
    else:
        return None
