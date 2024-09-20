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
        label_col='label'
    )

    # Assertions to check if the initialization is correct
    assert isinstance(fs_df, FSDataFrame)

    assert fs_df.get_sample_col_name() == 'sample_id'

def test_scaler_df():

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
        label_col='label'
    )

    # Scale the DataFrame
    fs_df.scale_features(scaler_method='standard')

    # Assertions to check if the scaling is correct
    assert fs_df.is_scaled() == True
    assert fs_df.get_scaled_method() == 'standard'