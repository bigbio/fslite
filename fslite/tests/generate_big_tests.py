import logging

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def generate_large_test_dataset():
    # Parameters for the dataset
    n_samples = 1200
    n_features = 10_000
    chunk_size = 100  # Adjust chunk size for memory efficiency

    # Generate sample IDs and labels
    sample_ids = np.arange(1, n_samples + 1)
    labels = np.random.choice(["LV", "RV", "LA", "RA"], size=n_samples)

    # Parquet schema definition
    schema = pa.schema(
        [pa.field("sample_id", pa.int32()), pa.field("label", pa.string())]
        + [pa.field(f"feature{i}", pa.float32()) for i in range(1, n_features + 1)]
    )

    # Create an empty Parquet file
    output_file = "large_dataset_optimized_samples_{}_features_{}.parquet".format(
        n_samples, n_features
    )
    with pq.ParquetWriter(output_file, schema, compression="snappy") as writer:
        # Process in chunks to reduce memory usage
        for chunk_start in range(0, n_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_samples)

            # Generate chunk of samples and labels
            chunk_sample_ids = sample_ids[chunk_start:chunk_end]
            chunk_labels = labels[chunk_start:chunk_end]

            # Generate chunk of features
            chunk_features = {
                f"feature{i}": np.random.rand(chunk_end - chunk_start)
                for i in range(1, n_features + 1)
            }

            # Create DataFrame chunk
            chunk_data = {"sample_id": chunk_sample_ids, "label": chunk_labels}
            chunk_data.update(chunk_features)

            df_chunk = pd.DataFrame(chunk_data)

            # Convert to PyArrow Table and write chunk to Parquet file
            table_chunk = pa.Table.from_pandas(df_chunk, schema=schema)
            writer.write_table(table_chunk)
            logging.info(f"Processed samples {chunk_start + 1} to {chunk_end}")

    print("Optimized Parquet file created successfully!")
