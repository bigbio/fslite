# Import and convert to parquet a single-cell dataset: GSE156793 (loom format)
# GEO URL:
# https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE156793&format=file&file=GSE156793%5FS3%5Fgene%5Fcount%2Eloom%2Egz

# import libraries
import pandas as pd
import loompy
import pyarrow.parquet as pq
import pyarrow as pa

# define the path to the loom file
loom_file = "GSE156793_S3_gene_count.loom"

# connect to the loom file
ds = loompy.connect(loom_file)

# get shape of the data
ds.shape

# retrieve the row attributes
ds.ra.keys()

# get gene ids
gene_ids = ds.ra["gene_id"]
gene_ids[0:10]

# get the column attributes
ds.ca.keys()

# get sample metadata
sample_id = ds.ca["sample"]
cell_cluster = ds.ca["Main_cluster_name"]
assay = ds.ca["Assay"]
development_day = ds.ca["Development_day"]

# make a dataframe with the sample metadata, define the columns types
sample_df = pd.DataFrame({
        "sample_id": sample_id,
        "cell_cluster": cell_cluster,
        "assay": assay,
        "development_day": development_day,
    }
)

# print the first 5 rows
sample_df.head()

# Make 'cell_cluster' a categorical variable encoded as an integer
sample_df["cell_cluster"] = sample_df["cell_cluster"].astype("category")
sample_df["cell_cluster_id"] = sample_df["cell_cluster"].cat.codes

# Make 'assay' a categorical variable encoded as an integer
sample_df["assay"] = sample_df["assay"].astype("category")
sample_df["assay_id"] = sample_df["assay"].cat.codes

# Make 'sample_id' the index
sample_df = sample_df.set_index("sample_id")

# Show the first 5 rows
sample_df.head()

# Save the sample metadata to parquet
(
    sample_df.reset_index().to_parquet(
        "sample_metadata.parquet", index=False, engine="auto", compression="gzip"
    )
)


# transpose dataset and convert to parquet.
# process the data per chunks.
chunk_size = 10000
writer = None
count = 0
number_chunks = 10 # number of chunks to process

for ix, selection, view in ds.scan(axis=1, batch_size=chunk_size):
    # retrieve the chunk
    matrix_chunk = view[:, :]

    # transpose the data
    matrix_chunk_t = matrix_chunk.T

    # convert to pandas dataframe
    df_chunk = pd.DataFrame(
        matrix_chunk_t, index=sample_id[selection.tolist()], columns=gene_ids
    )

    # merge chunk with sample metadata
    df_chunk = pd.merge(
        left=sample_df[["cell_cluster_id", "development_day", "assay_id"]],
        right=df_chunk,
        how="inner",
        left_index=True,
        right_index=True,
        sort=False,
        copy=True,
        indicator=False,
        validate="one_to_one",
    )

    # reset the index
    df_chunk = df_chunk.reset_index()

    # rename the index column
    df_chunk = df_chunk.rename(columns={"index": "sample_id"})

    if writer is None:
        # define the schema
        schema = pa.schema(
            [
                pa.field("sample_id", pa.string()),
                pa.field("cell_cluster_id", pa.int8()),
                pa.field("development_day", pa.int64()),
                pa.field("assay_id", pa.int8()),
            ]
            + [pa.field(gene_id, pa.float32()) for gene_id in gene_ids]
        )

        print(len(list(df_chunk.columns)))
        print(len(schema))

        # create the parquet writer
        writer = pq.ParquetWriter("GSE156793.parquet", schema, compression="snappy")

    writer.write_table(pa.Table.from_pandas(df_chunk, preserve_index=False))

    print(f"Chunk {ix} saved")

    count += 1
    if count >= number_chunks:
        break

if writer is not None:
    writer.close()
    print(f"Concatenated parquet file written to GSE156793.parquet")
