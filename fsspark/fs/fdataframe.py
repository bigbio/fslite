import logging
from typing import Optional, Union, List, Set, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

logging.basicConfig(format="%(levelname)s (%(name)s %(lineno)s): %(message)s")
logger = logging.getLogger("pickfeat")
logger.setLevel(logging.INFO)

class FSDataFrame:
    """
    FSDataFrame is a representation of a DataFrame with some functionalities to perform feature selection.
    An object from FSDataFrame is basically represented by a  DataFrame with samples
    as rows and features as columns, with extra distributed indexed pandas series for
    features names and samples labels.

    An object of FSDataFrame offers an interface to a DataFrame, a Pandas on  DataFrame
    (e.g. suitable for visualization) or a  DataFrame with features as a Dense column vector (e.g. suitable for
    applying most algorithms from  MLib API).

    It can also be split in training and testing dataset and filtered by removing selected features (by name or index).

    [...]

    """

    def __init__(
            self,
            df: DataFrame,
            sample_col: str = None,
            label_col: str = None,
            row_index_col: Optional[str] = '_row_index',
            parse_col_names: bool = False,
            parse_features: bool = False,
    ):
        """
        Create an instance of FSDataFrame.

        Expected an input DataFrame with 2+N columns.
        After specifying sample id and sample label columns, the remaining N columns will be considered as features.

        :param df: Pandas DataFrame
        :param sample_col: Sample id column name
        :param label_col: Sample label column name
        :param row_index_col: Optional. Column name of row indices.
        :param parse_col_names: Replace dots (.) in column names with underscores.
        :param parse_features: Coerce all features to float.
        """

        self.__df = self._convert_psdf_to_sdf(df)
        self.__sample_col = sample_col
        self.__label_col = label_col
        self.__row_index_name = row_index_col

        # check input dataframe
        self._check_df()

        # replace dots in column names, if any.
        if parse_col_names:
            #  TODO: Dots in column names are prone to errors, since dots are used to access attributes from DataFrame.
            #        Should we make this replacement optional? Or print out a warning?
            self.__df = self.__df.toDF(*(c.replace('.', '_') for c in self.__df.columns))

        # If the specified row index column name does not exist, add row index to the dataframe
        if self.__row_index_name not in self.__df.columns:
            self.__df = self._add_row_index(index_name=self.__row_index_name)

        if parse_features:
            # coerce all features to float
            non_features_cols = [self.__sample_col, self.__label_col, self.__row_index_name]
            feature_cols = [c for c in self.__df.columns if c not in non_features_cols]
            self.__df = self.__df.withColumns({c: self.__df[c].cast('float') for c in feature_cols})

        self.__indexed_features = self._set_indexed_cols()
        self.__indexed_instances = self._set_indexed_rows()

    def _check_df(self):
        """
        Check if input DataFrame meet the minimal requirements to feed an FS pipeline.

        :return: None
        """
        col_names = self.__df.columns
        if self.__sample_col not in col_names:
            raise ValueError(f"Column sample name {self.__sample_col} not found...")
        elif self.__label_col not in col_names:
            raise ValueError(f"Column label name {self.__label_col} not found...")
        elif not isinstance(self.__row_index_name, str):
            raise ValueError("Row index column name must be a valid string...")
        else:
            pass

    def _set_indexed_cols(self) -> Series:
        """
        Create a distributed indexed Series representing features.

        :return: Pandas on  (PoS) Series
        """
        # TODO: Check for equivalent to pandas distributed Series in .
        non_features_cols = [self.__sample_col, self.__label_col, self.__row_index_name]
        features = [f for f in self.__df.columns if f not in non_features_cols]
        return Series(features)

    def _set_indexed_rows(self) -> Series:
        """
        Create a distributed indexed Series representing samples labels.
        It will use existing row indices, if any.

        :return: Pandas on  (PoS) Series
        """
        # TODO: Check for equivalent to pandas distributed Series in .
        label = self.__df.select(self.__label_col).collect()
        row_index = self.__df.select(self.__row_index_name).collect()
        return Series(label, index=row_index)

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

        sdf = self.__df
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
        sdf = self.get_sdf().select(*self.get_features_names())
        a = np.array(sdf.collect())
        return a

    def to_psdf(self) -> DataFrame:
        """
        Convert  DataFrame to Pandas on DataFrame
        :return: Pandas on  DataFrame
        """
        return self.__df.pandas_api()

    def get_sdf(self) -> DataFrame:
       return self.__df

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
        return self.__row_index_name

    def _add_row_index(self, index_name: str = '_row_index') -> pd.DataFrame:
        """
        Add row indices to DataFrame.
        Unique indices of type integer will be added in non-consecutive increasing order.

        :param index_name: Name of the row index column.
        :return: DataFrame with extra column of row indices.
        """
        # Add a new column with unique row indices using a range
        self.__df[index_name] = range(len(self.__df))
        return self.__df

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
        sdf = self.get_sdf()

        if keep:
            sdf = sdf.select(
                self.__sample_col,
                self.__label_col,
                self.__row_index_name,
                *features)
        else:
            sdf = sdf.drop(*features)

        fsdf_filtered = self.update(sdf, self.__sample_col, self.__label_col, self.__row_index_name)
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

        features_col_vector = '_features'
        scaled_features_vector = '_features_scaled'

        sdf = self.get_sdf_vector(output_column_vector=features_col_vector)

        sdf = (scaler
               .setInputCol(features_col_vector)
               .setOutputCol(scaled_features_vector)
               .fit(sdf)
               .transform(sdf)
               .drop(features_col_vector)
               )

        sdf = _disassemble_column_vector(sdf,
                                         features_cols=self.get_features_names(),
                                         col_vector_name=scaled_features_vector,
                                         drop_col_vector=True)

        return self.update(sdf,
                           self.__sample_col,
                           self.__label_col,
                           self.__row_index_name)

    def split_df(self,
                 label_type_cat: bool = True,
                 split_training_factor: float = 0.7) -> Tuple['FSDataFrame', 'FSDataFrame']:
        """
        TODO: Split dataframe in training and test dataset, maintaining balance between classes.
        Split DataFrame into training and test dataset.
        It will generate a nearly class-balanced training
        and testing set for both categorical and continuous label input.

        :param label_type_cat: If True (the default), the input label colum will be processed as categorical.
                               Otherwise, it will be considered a continuous variable and binarized.
        :param split_training_factor: Proportion of the training set. Usually, a value between 0.6 and 0.8.

        :return: Tuple of FSDataFrames. First element is the training set and second element is the testing set.
        """




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

    def _assemble_column_vector(self,
                                input_feature_cols: List[str],
                                output_column_vector: str = 'features',
                                drop_input_cols: bool = True) -> pd.DataFrame:
        """
        Assemble features (columns) from DataFrame into a column of type Numpy array.

        :param drop_input_cols: Boolean flag to drop the input feature columns.
        :param input_feature_cols: List of feature column names.
        :param output_column_vector: Name of the output column that will contain the combined vector.
        :param sdf: Pandas DataFrame

        :return: DataFrame with column of type Numpy array.
        """

        # Combine the input columns into a single vector (Numpy array)
        self.__df[output_column_vector] = self.__df[input_feature_cols].apply(lambda row: np.array(row), axis=1)

        # Drop input columns if flag is set to True
        if drop_input_cols:
            return self.__df.drop(columns=input_feature_cols)
        else:
            return self.__df

def _disassemble_column_vector(self,
                                   features_cols: List[str],
                                   col_vector_name: str,
                                   drop_col_vector: bool = True) -> pd.DataFrame:
    """
    Convert a column of Numpy arrays in DataFrame to individual columns (a.k.a features).
    This is the reverse operation of `_assemble_column_vector`.

    :param features_cols: List of new feature column names.
    :param col_vector_name: Name of the column that contains the vector (Numpy array).
    :param drop_col_vector: Boolean flag to drop the original vector column.
    :return: DataFrame with individual feature columns.
    """

    # Unpack the vector (Numpy array) into individual columns
    for i, feature in enumerate(features_cols):
        self.__df[feature] = self.__df[col_vector_name].apply(lambda x: x[i])

    # Drop the original vector column if needed
    if drop_col_vector:
       self.__df = self.__df.drop(columns=[col_vector_name])

    return self.__df

