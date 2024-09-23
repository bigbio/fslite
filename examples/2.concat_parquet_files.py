import pyspark
from pyspark.sql import SparkSession

# create spark session
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("fsspark") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \
    .config("spark.sql.pivotMaxValues", "100000") \
    .config("spark.network.timeout", "100000") \
    .config("spark.sql.session.timeZone", "UTC") \
    .config("spark.executor.memory", "80g") \
    .config("spark.driver.memory", "100g") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "8g") \
    .config("spark.sql.session.timeZone", "UTC") \
    .getOrCreate()


# get all absolute paths of files in a directory
def get_files_paths(directory, extension: str = "parquet.gz"):
    """
    Get all files paths in a directory.
    :param extension: str, file extension.
    :param directory: str, directory path.
    :return: list, list of files paths.
    """
    import os
    files_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                files_paths.append(os.path.join(root, file))
    return files_paths


# get all files paths
files_paths = get_files_paths(directory="/mnt/nfs/user-data/eam/GSE156793/sdf",
                              extension="parquet.gz")

# read all parquet files as spark dataframe and write (append) them to a single parquet file
for file_path in files_paths:
    print("Processing file: {}".format(file_path))
    df = (spark
          .read
          .parquet(file_path)
          .repartition(5)
          )

    df.write.parquet("/mnt/nfs/user-data/eam/GSE156793/GSE156793.sample_x_gene.parquet",
                     mode='append',
                     compression='gzip')