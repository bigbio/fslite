import logging
from typing import List, Tuple, Optional, Union

import numpy
import numpy as np
import pandas as pd
import psutil
from pandas import DataFrame
from scipy import sparse
from sklearn.preprocessing import (
    MinMaxScaler,
    MaxAbsScaler,
    StandardScaler,
    RobustScaler,
    LabelEncoder,
)

logging.basicConfig(format="%(levelname)s (%(name)s %(lineno)s): %(message)s")
logger = logging.getLogger("pickfeat")
logger.setLevel(logging.INFO)


class FSDataFrame:
    """
    FSDataFrame is a representation of a DataFrame with some functionalities to perform feature selection.
    An object from FSDataFrame is basically represented by a DataFrame with samples
    as rows and features as columns, with extra distributed indexed pandas series for
    feature names and samples labels.

    An object of FSDataFrame offers an interface to a DataFrame, a Pandas on  DataFrame
    (e.g., suitable for visualization) or a DataFrame with features as a Dense column vector (e.g. suitable for
    applying most algorithms from MLib API).

    It can also be split in training and testing dataset and filtered by removing selected features (by name or index).

    [...]

    """

    def __init__(
        self,
        df: pd.DataFrame,
        sample_col: Optional[str] = None,
        label_col: Optional[str] = None,
        sparse_threshold: float = 0.7,  # Threshold for sparsity
        memory_threshold: Optional[
            float
        ] = 0.75,  # Proportion of system memory to use for dense arrays
    ):
        """
        Create an instance of FSDataFrame.

        The input DataFrame should contain 2+N columns. After specifying the sample id and label columns,
        the remaining N columns will be considered features. The feature columns should contain only numerical data.
        The DataFrame is stored in a dense or sparse format based on the sparsity of the data and available memory.

        :param df: Input Pandas DataFrame
        :param sample_col: Column name for sample identifiers (optional)
        :param label_col: Column name for labels (required)
        :param sparse_threshold: Threshold for sparsity, default is 70%. If the proportion of zero entries
        in the feature matrix exceeds this value, the matrix is stored in a sparse format unless memory allows.
        :param memory_threshold: Proportion of system memory available to use before deciding on sparse/dense.
        """
        self.__df = df.copy()

        # Check for necessary columns
        columns_to_drop = []

        # Handle sample column
        if sample_col:
            if sample_col not in df.columns:
                raise ValueError(
                    f"Sample column '{sample_col}' not found in DataFrame."
                )
            self.__sample_col = sample_col
            self.__samples = df[sample_col].tolist()
            columns_to_drop.append(sample_col)
        else:
            self.__sample_col = None
            self.__samples = []
            logging.info("No sample column specified.")

        # Handle label column
        if label_col is None:
            raise ValueError("A label column is required but was not specified.")
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in DataFrame.")

        self.__label_col = label_col
        self.__labels = df[label_col].tolist()

        # Encode labels
        label_encoder = LabelEncoder()
        self.__labels_matrix = label_encoder.fit_transform(df[label_col]).tolist()
        columns_to_drop.append(label_col)

        # Drop both sample and label columns in one step
        self.__df = self.__df.drop(columns=columns_to_drop)

        # Extract features
        self.__original_features = self.__df.columns.tolist()

        # Ensure only numerical features are retained
        numerical_df = self.__df.select_dtypes(include=[np.number])
        if numerical_df.empty:
            raise ValueError("No numerical features found in the DataFrame.")

        # Check sparsity
        num_elements = numerical_df.size
        num_zeros = (numerical_df == 0).sum().sum()
        sparsity = num_zeros / num_elements

        dense_matrix_size = numerical_df.memory_usage(deep=True).sum()  # In bytes
        available_memory = psutil.virtual_memory().available  # In bytes

        if sparsity > sparse_threshold:
            if dense_matrix_size < memory_threshold * available_memory:
                # Use dense matrix if enough memory is available
                logging.info(
                    f"Data is sparse (sparsity={sparsity:.2f}) but enough memory available. "
                    f"Using a dense matrix."
                )
                self.__matrix = numerical_df.to_numpy(dtype=np.float32)
                self.__is_sparse = False
            else:
                # Use sparse matrix due to memory constraints
                logging.info(
                    f"Data is sparse (sparsity={sparsity:.2f}), memory insufficient for dense matrix. "
                    f"Using a sparse matrix representation."
                )
                self.__matrix = sparse.csr_matrix(
                    numerical_df.to_numpy(dtype=np.float32)
                )
                self.__is_sparse = True
        else:
            # Use dense matrix since it's not sparse
            logging.info(
                f"Data is not sparse (sparsity={sparsity:.2f}), using a dense matrix."
            )
            self.__matrix = numerical_df.to_numpy(dtype=np.float32)
            self.__is_sparse = False

        self.__is_scaled = (False, None)

    def get_feature_matrix(self) -> Union[np.ndarray, sparse.csr_matrix]:
        return self.__matrix

    def get_label_vector(self) -> numpy.array:
        return self.__labels_matrix

    def get_sample_col_name(self) -> str:
        """
        Return sample id column name.

        :return: Sample id column name.
        """
        return self.__sample_col

    def get_label_col_name(self) -> str:
        """
        Return sample label column name.

        :return: Sample label column name.
        """
        return self.__label_col

    def count_features(self) -> int:
        """
        Return the number of features.
        :return: Number of features.
        """
        return self.__matrix.shape[1]

    def count_instances(self) -> int:
        """
        Return the number of samples (instances).
        :return: Number of samples.
        """
        return self.__matrix.shape[0]

    def scale_features(self, scaler_method: str = "standard", **kwargs) -> bool:
        """
        Scales features in the SDataFrame using a specified method.

        :param scaler_method: One of: min_max, max_abs, standard or robust.
        :return: FSDataFrame with scaled features.
        """

        if scaler_method == "min_max":
            scaler = MinMaxScaler(**kwargs)
        elif scaler_method == "max_abs":
            scaler = MaxAbsScaler(**kwargs)
        elif scaler_method == "standard":
            scaler = StandardScaler(**kwargs)
        elif scaler_method == "robust":
            scaler = RobustScaler(**kwargs)
        else:
            raise ValueError(
                "`scaler_method` must be one of: min_max, max_abs, standard or robust."
            )

        # TODO: Scale only the features for now, we have to investigate if we scale categorical variables
        self.__matrix = scaler.fit_transform(self.__matrix)
        self.__is_scaled = (True, scaler_method)
        return True

    def is_scaled(self):
        return self.__is_scaled[0]

    def get_scaled_method(self):
        return self.__is_scaled[1]

    def is_sparse(self):
        return self.__is_sparse

    def select_features_by_index(self, feature_indexes: List[int]) -> "FSDataFrame":
        """
        Keep only the specified features (by index) and return an updated instance of FSDataFrame.

        :param feature_indexes: List of feature column indices to keep.
        :return: A new FSDataFrame instance with only the selected features.
        """
        # Filter the feature matrix to retain only the selected columns (features)
        updated_matrix = self.__matrix[:, feature_indexes]

        # Filter the original feature names to retain only the selected ones
        updated_features = [self.__original_features[i] for i in feature_indexes]

        # Create a new DataFrame with the retained features and their names
        updated_df = pd.DataFrame(updated_matrix, columns=updated_features)

        # Reattach the sample column (if it exists)
        if self.__sample_col:
            updated_df[self.__sample_col] = self.__samples

        # Reattach the label column
        updated_df[self.__label_col] = self.__labels

        # Return a new instance of FSDataFrame with the updated data
        return FSDataFrame(
            updated_df, sample_col=self.__sample_col, label_col=self.__label_col
        )

    def to_pandas(self) -> DataFrame:
        """
        Return the DataFrame representation of the FSDataFrame.
        :return: Pandas DataFrame.
        """

        df = pd.DataFrame()

        # Reattach the sample column (if it exists)
        if self.__sample_col:
            df[self.__sample_col] = self.__samples

        # Reattach the label column
        df[self.__label_col] = self.__labels

        # Create a DataFrame from the feature matrix
        df_features = pd.DataFrame(self.__matrix, columns=self.__original_features)

        # Concatenate the features DataFrame
        df = pd.concat([df, df_features], axis=1)

        return df

    def split_df(
        self, label_type_cat: bool = True, split_training_factor: float = 0.7
    ) -> Tuple["FSDataFrame", "FSDataFrame"]:
        """
        Split DataFrame into training and test dataset.
        It will generate a nearly class-balanced training
        and testing set for both categorical and continuous label input.

        :param label_type_cat: If True (the default), the input label column will be processed as categorical.
                               Otherwise, it will be considered a continuous variable and binarized.
        :param split_training_factor: Proportion of the training set. Usually, a value between 0.6 and 0.8.

        :return: Tuple of FSDataFrames. First element is the training set and second element is the testing set.
        //TODO: To be done.
        """

        # label_col = self.get_label_col_name()
        # df = self.__matrix.copy()
        #
        # # Create a temporary label column for sampling
        # tmp_label_col = '_tmp_label_indexed'
        #
        # if label_type_cat:
        #     # Use factorize to convert categorical labels to integer indices
        #     df[tmp_label_col], _ = pd.factorize(df[label_col])
        # else:
        #     # For continuous labels, create a uniform random column and binarize it
        #     df['_tmp_uniform_rand'] = np.random.rand(len(df))
        #     df[tmp_label_col] = (df['_tmp_uniform_rand'] > 0.5).astype(int)
        #     df = df.drop(columns=['_tmp_uniform_rand'])
        #
        # # Perform stratified sampling to get class-balanced training set
        # train_df = df.groupby(tmp_label_col, group_keys=False).apply(lambda x: x.sample(frac=split_training_factor))
        #
        # # Get the test set by subtracting the training set from the original DataFrame
        # test_df = df.drop(train_df.index)
        #
        # # Drop the temporary label column
        # train_df = train_df.drop(columns=[tmp_label_col])
        # test_df = test_df.drop(columns=[tmp_label_col])
        #
        # # Return the updated DataFrames
        # return self.update(train_df), self.update(test_df)
