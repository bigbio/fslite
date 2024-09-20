import logging
from typing import Optional, Union, List, Set, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, LabelEncoder

logging.basicConfig(format="%(levelname)s (%(name)s %(lineno)s): %(message)s")
logger = logging.getLogger("pickfeat")
logger.setLevel(logging.INFO)


class FSDataFrame:
    """
    FSDataFrame is a representation of a DataFrame with some functionalities to perform feature selection.
    An object from FSDataFrame is basically represented by a DataFrame with samples
    as rows and features as columns, with extra distributed indexed pandas series for
    features names and samples labels.

    An object of FSDataFrame offers an interface to a DataFrame, a Pandas on  DataFrame
    (e.g. suitable for visualization) or a  DataFrame with features as a Dense column vector (e.g. suitable for
    applying most algorithms from MLib API).

    It can also be split in training and testing dataset and filtered by removing selected features (by name or index).

    [...]

    """

    def __init__(
            self,
            df: DataFrame,
            sample_col: str = None,
            label_col: str = None,
            row_index_col: Optional[str] = '_row_index',
    ):
        """
        Create an instance of FSDataFrame.

        Expected an input DataFrame with 2+N columns.
        After specifying sample id and sample label columns, the remaining N columns will be considered as features.

        :param df: Pandas DataFrame
        :param sample_col: Sample id column name
        :param label_col: Sample label column name
        :param row_index_col: Optional. Column name of row indices.
        """

        if sample_col is None:
            self.__sample_col = None
            self.__samples = []
            logging.info("No sample column specified.")
        else:
            self.__sample_col = sample_col
            self.__samples = df[sample_col].tolist()
            df = df.drop(columns=[sample_col])

        if label_col is None:
            raise ValueError("No label column specified. A class/label column is required.")
        else:
            self.__label_col = label_col
            self.__labels = df[label_col].tolist()
            label_encoder = LabelEncoder()
            self.__labels_matrix = label_encoder.fit_transform(df[label_col]).tolist()
            df = df.drop(columns=[label_col])

        self.__original_features = df.columns.tolist()
        numerical_df = df.select_dtypes(include=[np.number])
        self.__matrix = numerical_df.to_numpy(dtype=np.float32)

    def _set_indexed_cols(self) -> Series:
        """
        Create a distributed indexed Series representing features.
        :return: Pandas on  (PoS) Series
        """
        non_features_cols = [self.__sample_col, self.__label_col, self.__row_index_col]
        features = [f for f in self.__matrix.columns if f not in non_features_cols]
        return Series(features)

    def _set_indexed_rows(self) -> pd.Series:
        """
        Create an indexed Series representing sample labels.
        It will use existing row indices from the DataFrame.

        :return: Pandas Series
        """

        # Extract the label and row index columns from the DataFrame
        labels = self.__matrix[self.__label_col]
        row_indices = self.__matrix[self.__row_index_col]

        # Create a Pandas Series with row_indices as index and labels as values
        return pd.Series(data=labels.values, index=row_indices.values)

    def get_features_indexed(self) -> Series:
        """
        Return features names with indices as a Series.
        :return: Indexed Series.
        """
        return self.__indexed_features

    def get_sample_label_indexed(self) -> Series:
        """
        Return sample labels with indices as a Series.
        :return: Indexed Series.
        """
        return self.__indexed_instances

    def get_features_names(self) -> list:
        """
        Get features names from DataFrame.
        :return: List of features names
        """
        return self.__indexed_features.tolist()

    def get_features_by_index(self, indices: Union[List[int], Set[int]]) -> List[str]:
        """
        Get features names by specified index from DataFrame.

        :param: indices: List of feature indexes
        :return: List of features names
        """
        return self.__indexed_features.loc[indices].tolist()

    def get_sample_label(self) -> list:
        """
        Get samples class (label) from DataFrame.
        :return: List of sample class labels
        """
        return self.__indexed_instances.tolist()

    def get_sdf_vector(self, output_column_vector: str = 'features') -> pd.DataFrame:
        """
        Return a  dataframe with feature columns assembled into a column vector (a.k.a. Dense Vector column).
        This format is required as input for multiple algorithms from MLlib API.

        :param: output_column_vector: Name of the output column vector.
        :return:  DataFrame
        """

        sdf = self.__matrix
        features_cols = self.get_features_names()
        sdf_vector = _assemble_column_vector(sdf,
                                             input_feature_cols=features_cols,
                                             output_column_vector=output_column_vector)

        return sdf_vector

    def get_sdf_and_label(self,
                          output_column_vector: str = 'features') -> Tuple[DataFrame, str, str]:
        """
        Extracts the  DataFrame and label column name from FSDataFrame.

        :param: output_column_vector: Name of the output column vector.
        :return: A tuple containing the  DataFrame and the label column name.
        """
        sdf = self.get_sdf_vector(output_column_vector=output_column_vector)
        label_col = self.get_label_col_name()
        return sdf, label_col, output_column_vector

    def _collect_features_as_array(self) -> np.array:
        """
        Collect features from FSDataFrame as an array.
        `Warning`: This method will collect the entire DataFrame into the driver.
                   Uses this method on small datasets only (e.g., after filtering or splitting the data)

        :return: Numpy array
        """
        sdf = self.get_df().select(*self.get_features_names())
        a = np.array(sdf.collect())
        return a

    def to_psdf(self) -> DataFrame:
        """
        Convert  DataFrame to Pandas on DataFrame
        :return: Pandas on  DataFrame
        """
        return self.__matrix.pandas_api()

    def get_df(self) -> DataFrame:
        return self.__matrix

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

    def get_row_index_name(self) -> str:
        """
        Return row (instances) id column name.

        :return: Row id column name.
        """
        return self.__row_index_col

    def _add_row_index(self, index_name: str = '_row_index') -> pd.DataFrame:
        """
        Add row indices to DataFrame.
        Unique indices of type integer will be added in non-consecutive increasing order.

        :param index_name: Name of the row index column.
        :return: DataFrame with extra column of row indices.
        """
        # Add a new column with unique row indices using a range
        self.__matrix[index_name] = list(range(len(self.__matrix)))
        return self.__matrix

    def count_features(self) -> int:
        """
        Return the number of features.

        :return: Number of features.
        """
        return self.get_features_indexed().size

    def count_instances(self) -> int:
        """
        Return the number of samples (instances).

        :return: Number of samples.
        """
        return self.get_sample_label_indexed().size

    def filter_features(self, features: List[str], keep: bool = True) -> 'FSDataFrame':
        """
        Select or drop specified features from DataFrame.

        :param features: List of features names to drop or select from DataFrame
        :param keep: If True (default), keep features. Remove otherwise.

        :return: FSDataFrame
        """

        current_features = self.get_features_names()
        if len(set(current_features).intersection(features)) == 0:
            logger.warning(f"There is no overlap of specified features with the input data frame.\n"
                           f"Skipping this filter step...")
            return self

        count_a = self.count_features()
        sdf = self.get_df()

        if keep:
            sdf = sdf.select(
                self.__sample_col,
                self.__label_col,
                self.__row_index_col,
                *features)
        else:
            sdf = sdf.drop(*features)

        fsdf_filtered = self.update(sdf, self.__sample_col, self.__label_col, self.__row_index_col)
        count_b = fsdf_filtered.count_features()

        logger.info(f"{count_b} features out of {count_a} remain after applying this filter...")

        return fsdf_filtered

    def filter_features_by_index(self, feature_indices: Set[int], keep: bool = True) -> 'FSDataFrame':
        """
        Select or drop specified features from DataFrame by its indices.

        :param feature_indices: Set of features indices to drop or select from DataFrame
        :param keep: If True (default), keep features. Remove otherwise.

        :return: FSDataFrame
        """
        feature_names = self.get_features_by_index(feature_indices)
        return self.filter_features(feature_names, keep=keep)

    def get_label_strata(self) -> list:
        """
        Get strata from a categorical column in DataFrame.

        :return: List of levels for categorical variable.
        """
        levels = self.get_sample_label_indexed().unique().tolist()
        number_of_lvs = len(levels)
        if number_of_lvs > 20:  # TODO: Check if this is a right cutoff.
            logger.warning(f"Number of observed levels too high: {number_of_lvs}.\n"
                           f"Should this variable be considered continuous?")
        return levels

    def scale_features(self, scaler_method: str = 'standard', **kwargs) -> 'FSDataFrame':
        """
        Scales features in DataFrame

        :param scaler_method: One of: min_max, max_abs, standard or robust.
        :return: FSDataFrame with scaled features.
        """

        if scaler_method == 'min_max':
            scaler = MinMaxScaler(**kwargs)
        elif scaler_method == 'max_abs':
            scaler = MaxAbsScaler(**kwargs)
        elif scaler_method == 'standard':
            scaler = StandardScaler(**kwargs)
        elif scaler_method == 'robust':
            scaler = RobustScaler(**kwargs)
        else:
            raise ValueError("`scaler_method` must be one of: min_max, max_abs, standard or robust.")

        feature_array = self._features_to_array()

        feature_array = (scaler
                         .fit(feature_array)
                         .transform()
                         )

        df_scaled = self._array_to_features(feature_array)

        return self.update(sdf,
                           self.__sample_col,
                           self.__label_col,
                           self.__row_index_name)

    def split_df(self,
                 label_type_cat: bool = True,
                 split_training_factor: float = 0.7) -> Tuple['FSDataFrame', 'FSDataFrame']:
        """
        Split DataFrame into training and test dataset.
        It will generate a nearly class-balanced training
        and testing set for both categorical and continuous label input.

        :param label_type_cat: If True (the default), the input label column will be processed as categorical.
                               Otherwise, it will be considered a continuous variable and binarized.
        :param split_training_factor: Proportion of the training set. Usually, a value between 0.6 and 0.8.

        :return: Tuple of FSDataFrames. First element is the training set and second element is the testing set.
        """

        label_col = self.get_label_col_name()
        df = self.__matrix.copy()

        # Create a temporary label column for sampling
        tmp_label_col = '_tmp_label_indexed'

        if label_type_cat:
            # Use factorize to convert categorical labels to integer indices
            df[tmp_label_col], _ = pd.factorize(df[label_col])
        else:
            # For continuous labels, create a uniform random column and binarize it
            df['_tmp_uniform_rand'] = np.random.rand(len(df))
            df[tmp_label_col] = (df['_tmp_uniform_rand'] > 0.5).astype(int)
            df = df.drop(columns=['_tmp_uniform_rand'])

        # Perform stratified sampling to get class-balanced training set
        train_df = df.groupby(tmp_label_col, group_keys=False).apply(lambda x: x.sample(frac=split_training_factor))

        # Get the test set by subtracting the training set from the original DataFrame
        test_df = df.drop(train_df.index)

        # Drop the temporary label column
        train_df = train_df.drop(columns=[tmp_label_col])
        test_df = test_df.drop(columns=[tmp_label_col])

        # Return the updated DataFrames
        return self.update(train_df), self.update(test_df)

    @classmethod
    def update(cls,
               df: DataFrame,
               sample_col: str,
               label_col: str,
               row_index_col: str):
        """
        Create a new instance of FSDataFrame.

        :param df:  DataFrame
        :param sample_col: Name of sample id column.
        :param label_col: Name of sample label column.
        :param row_index_col: Name of row (instances) id column.

        :return: FSDataFrame
        """
        return cls(df, sample_col, label_col, row_index_col)

    def _features_to_array(self) -> np.array:
        """
        Collect features from FSDataFrame as an array.
        `Warning`: This method will collect the entire DataFrame into the driver.
                   Uses this method on small datasets only (e.g., after filtering or splitting the data)

        :return: Numpy array
        """
        sdf = self.get_df().select(*self.get_features_names())
        a = np.array(sdf.collect())
        return a

    def _array_to_features(self, a: np.array) -> pd.DataFrame:
        """
        Convert a Numpy array to a DataFrame with features as columns.
        :param a: Numpy array
        :return: Pandas DataFrame
        """
        return pd.DataFrame(a, columns=self.get_features_names())

#
# def _assemble_column_vector(self,
#                             input_feature_cols: List[str],
#                             output_column_vector: str = 'features',
#                             drop_input_cols: bool = True) -> pd.DataFrame:
#     """
#     Assemble features (columns) from DataFrame into a column of type Numpy array.
#
#     :param drop_input_cols: Boolean flag to drop the input feature columns.
#     :param input_feature_cols: List of feature column names.
#     :param output_column_vector: Name of the output column that will contain the combined vector.
#     :param sdf: Pandas DataFrame
#
#     :return: DataFrame with column of type Numpy array.
#     """
#
#     # Combine the input columns into a single vector (Numpy array)
#     self.__df[output_column_vector] = self.__df[input_feature_cols].apply(lambda row: np.array(row), axis=1)
#
#     # Drop input columns if flag is set to True
#     if drop_input_cols:
#         return self.__df.drop(columns=input_feature_cols)
#     else:
#         return self.__df
#
#
# def _disassemble_column_vector(self,
#                                features_cols: List[str],
#                                col_vector_name: str,
#                                drop_col_vector: bool = True) -> pd.DataFrame:
#     """
#     Convert a column of Numpy arrays in DataFrame to individual columns (a.k.a features).
#     This is the reverse operation of `_assemble_column_vector`.
#
#     :param features_cols: List of new feature column names.
#     :param col_vector_name: Name of the column that contains the vector (Numpy array).
#     :param drop_col_vector: Boolean flag to drop the original vector column.
#     :return: DataFrame with individual feature columns.
#     """
#
#     # Unpack the vector (Numpy array) into individual columns
#     for i, feature in enumerate(features_cols):
#         self.__df[feature] = self.__df[col_vector_name].apply(lambda x: x[i])
#
#     # Drop the original vector column if needed
#     if drop_col_vector:
#         self.__df = self.__df.drop(columns=[col_vector_name])
#
#     return self.__df
