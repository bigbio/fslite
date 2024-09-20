import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
import gc

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

def test_memory_fsdataframe():
    def create_test_data(n_samples, n_features):
        """Create test data for FSDataFrame."""
        data = np.random.rand(n_samples, n_features)
        df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)])
        df['sample_id'] = [f'sample_{i}' for i in range(n_samples)]
        df['label'] = np.random.choice(['A', 'B'], n_samples)
        return df

    def measure_memory_usage(n_samples, n_features):
        """Measure memory usage of FSDataFrame for given number of samples and features."""
        df = create_test_data(n_samples, n_features)
        mem_usage = memory_usage((FSDataFrame, (df, 'sample_id', 'label')), max_iterations=1)[0]
        gc.collect()  # Force garbage collection to free memory
        return mem_usage

    # Define test cases
    feature_sizes = [1000, 5000, 10000, 50000, 100_000, 1_000_000]
    sample_sizes = [10, 50, 100, 500, 1000]

    # Measure memory usage for each test case
    results = []
    for n_samples in sample_sizes:
        for n_features in feature_sizes:
            mem_usage = measure_memory_usage(n_samples, n_features)
            results.append((n_samples, n_features, mem_usage))

    # Convert results to DataFrame
    results_df = pd.DataFrame(results, columns=['Samples', 'Features', 'Memory (MB)'])

    # Create 2D line plot
    plt.figure(figsize=(12, 8))

    for feature_size in feature_sizes:
        data = results_df[results_df['Features'] == feature_size]
        plt.plot(data['Samples'], data['Memory (MB)'], marker='o', label=f'{feature_size} Features')

    plt.xlabel('Number of Samples')
    plt.ylabel('Memory Usage (MB)')
    plt.title('FSDataFrame Memory Usage')
    plt.legend()
    plt.xscale('log')  # Using log scale for x-axis to better visualize the range
    plt.tight_layout()
    plt.show()

    # Print results table
    print(results_df.to_string(index=False))