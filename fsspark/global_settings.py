# some useful extra spark settings to test this module locally.
SPARK_EXTRA_SETTINGS = {'spark.executor.memory': '8g',
                        'spark.driver.memory': '20g',
                        "spark.memory.offHeap.enabled": 'true',
                        "spark.memory.offHeap.size": '2g',
                        "spark.sql.pivotMaxValues": '60000',
                        "spark.network.timeout": '100000'
                        }

# pyarrow settings to make available columnar data processing
PYARROW_SETTINGS = {"spark.sql.execution.arrow.pyspark.enabled": "true",
                    "spark.sql.execution.arrow.pyspark.fallback.enabled": "true"
                    }

# setting for pandas api on spark
PANDAS_ON_SPARK_API_SETTINGS = {"compute.default_index_type": "distributed",
                                "compute.ordered_head": False,
                                "display.max_rows": 100}
