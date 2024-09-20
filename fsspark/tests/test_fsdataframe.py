import pytest
import pandas as pd
from fsspark.fs.fdataframe import FSDataFrame

def test_initializes_fsdataframe():

    # Create a sample DataFrame
    data = {
        'sample_id': [1, 2, 3],
        'label': ['A', 'B', 'C'],
        'feature1': [0.1, 0.2, 0.3],
        'feature2': [1.1, 1.2, 1.3]
    }
    df = pd.DataFrame(data)

    # Initialize FSDataFrame
    fs_df = FSDataFrame(
        df=df,
        sample_col='sample_id',
        label_col='label',
        row_index_col='_row_index',
    )

    # Assertions to check if the initialization is correct
    assert isinstance(fs_df, FSDataFrame)