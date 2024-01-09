# Description: Global settings for the fsspark package.
# These settings provide a way to configure the spark session and the spark context to run the fsspark package locally.

# spark settings to test this module locally.
SPARK_EXTRA_SETTINGS = {'spark.executor.memory': '8g',
                        'spark.driver.memory': '20g',
                        "spark.memory.offHeap.enabled": 'true',
                        "spark.memory.offHeap.size": '2g',
                        "spark.sql.pivotMaxValues": '60000',
                        "spark.network.timeout": '100000',
                        "spark.sql.session.timeZone": "UTC"
                        }

# pyarrow settings to make available columnar data processing
PYARROW_SETTINGS = {"spark.sql.execution.arrow.pyspark.enabled": "true",
                    "spark.sql.execution.arrow.pyspark.fallback.enabled": "true"
                    }

# setting for pandas api on spark (PoS)
PANDAS_ON_SPARK_API_SETTINGS = {"compute.default_index_type": "distributed",
                                "compute.ordered_head": False,
                                "display.max_rows": 100}
