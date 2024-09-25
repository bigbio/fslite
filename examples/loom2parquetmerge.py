import os
import pyarrow.parquet as pq
import pyarrow as pa


# get all absolute paths of files in a directory
def get_files_paths(directory, extension: str = "parquet"):
    """
    Get all file paths in a directory.
    :param extension: str, file extension.
    :param directory: str, directory path.
    :return: list, list of file paths.
    """
    files_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                files_paths.append(os.path.join(root, file))
    return files_paths


def concatenate_parquet_files_incremental(files_paths, output_path, batch_size=10000):
    """
    Concatenate multiple parquet files in an incremental fashion to avoid memory overload.

    :param files_paths: List of parquet file paths.
    :param output_path: Path to the output parquet file.
    :param batch_size: Number of rows to read from each file at a time.
    """
    writer = None
with pq.ParquetWriter(output_path, schema=None, compression='gzip') as writer:
    for file_path in files_paths:
        print(f"Processing file: {file_path}")
        parquet_file = pq.ParquetFile(file_path)

        # Read the file in batches to avoid memory overload
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            # Convert the batch to a PyArrow Table
            table = pa.Table.from_batches([batch])

            # Write the batch to the output Parquet file
            writer.write_table(table)

print(f"Concatenated parquet file written to {output_path}")
        print(f"Concatenated parquet file written to {output_path}")


# Get all files paths
files_paths = get_files_paths(directory="./",
                              extension="parquet")

# Output path for the final concatenated parquet file
output_path = "GSE156793.parquet"

# Concatenate the parquet files and write to a single file incrementally
concatenate_parquet_files_incremental(files_paths, output_path, batch_size=10000)
