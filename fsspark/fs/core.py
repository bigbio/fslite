import logging
from typing import (Union,
                    Optional,
                    List,
                    Set,
                    Tuple)

import pyspark.pandas
import pyspark.sql
from pyspark.ml.feature import (VectorAssembler,
                                StringIndexer,
                                Binarizer,
                                MinMaxScaler,
                                MaxAbsScaler,
                                StandardScaler,
                                RobustScaler)
from pyspark.ml.functions import vector_to_array
from pyspark.pandas.series import Series
from pyspark.sql.functions import (monotonically_increasing_id,
                                   col,
                                   rand)

logging.basicConfig(format="%(levelname)s (%(name)s %(lineno)s): %(message)s")
logger = logging.getLogger("FSSPARK")
logger.setLevel(logging.INFO)


class FSDataFrame:
    """
    FSDataFrame is a representation of a Spark DataFrame with some functionalities to perform feature selection.
    An object from FSDataFrame is basically represented by a Spark DataFrame with samples
    as rows and features as columns, with extra distributed indexed pandas series for
    features names and samples labels.

    An object of FSDataFrame offers an interface to a Spark DataFrame, a Pandas on Spark DataFrame
    (e.g. suitable for visualization) or a Spark DataFrame with features as a Dense column vector (e.g. suitable for
    applying most algorithms from Spark MLib API).

    It can also be split in training and testing dataset and filtered by removing selected features (by name or index).

    [...]

    """

    def __init__(
            self,
            df: Union[pyspark.sql.DataFrame, pyspark.pandas.DataFrame],
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

        :param df: Spark (or Pandas on Spark) DataFrame
        :param sample_col: Sample id column name
        :param label_col: Sample label column name
        :param row_index_col: Optional. Column name of row indices.
        :param parse_col_names:
        :param parse_features:
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
            raise DataFormatError(f"Column sample name {self.__sample_col} not found...")
        elif self.__label_col not in col_names:
            raise DataFormatError(f"Column label name {self.__label_col} not found...")
        elif not isinstance(self.__row_index_name, str):
            raise DataFormatError("Row index column name must be a valid string...")
        else:
            pass

    @staticmethod
    def _convert_psdf_to_sdf(df: Union[pyspark.pandas.DataFrame, pyspark.sql.DataFrame]) -> pyspark.sql.DataFrame:
        """
        Convert Pandas on Spark DataFrame (psdf) to Spark DataFrame (sdf).
        :return: Spark DataFrame
        """
        return df.to_spark(index_col=None) if isinstance(df, pyspark.pandas.DataFrame) else df

    def _set_indexed_cols(self) -> pyspark.pandas.series.Series:
        """
        Create a distributed indexed Series representing features.

        :return: Pandas on Spark (PoS) Series
        """
        # TODO: Check for equivalent to pandas distributed Series in Spark.
        non_features_cols = [self.__sample_col, self.__label_col, self.__row_index_name]
        features = [f for f in self.__df.columns if f not in non_features_cols]
        return Series(features)

    def _set_indexed_rows(self) -> pyspark.pandas.series.Series:
        """
        Create a distributed indexed Series representing samples labels.
        It will use already existing row indices.

        :return: Pandas on Spark (PoS) Series
        """
        # TODO: Check for equivalent to pandas distributed Series in Spark.
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
        :return: List of features names
        """
        return self.__indexed_features.loc[indices].tolist()

    def get_sample_label(self) -> list:
        """
        Get samples class (label) from DataFrame.
        :return: Pandas Series
        """
        return self.__indexed_instances.tolist()

    # def get_samples(self) -> pyspark.pandas.Series:
    #     """
    #     Get samples identifiers from DataFrame. Coerce data type to string.
    #
    #     :return: Pandas Series
    #     """
    #     return self.__df[self.__sample_col].astype("str")

    def get_sdf_vector(self, output_column_vector: str = 'features') -> pyspark.sql.DataFrame:
        """
        Return a Spark dataframe with feature columns assembled into a column vector (a.k.a. Dense Vector column).
        This format is required as input for multiple algorithms from MLlib API.

        :return: Spark DataFrame
        """

        sdf = self.__df
        features_cols = self.get_features_names()
        sdf_vector = _assemble_column_vector(sdf,
                                             input_feature_cols=features_cols,
                                             output_column_vector=output_column_vector)

        return sdf_vector

    def to_psdf(self) -> pyspark.pandas.DataFrame:
        """
        Convert Spark DataFrame to Pandas on Spark DataFrame
        :return: Pandas on Spark DataFrame
        """
        return self.__df.pandas_api()

    def get_sdf(self) -> pyspark.sql.DataFrame:
        """
        Return current Spark DataFrame
        :return: Spark DataFrame
        """
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

    def _add_row_index(self, index_name: str = '_row_index') -> pyspark.sql.DataFrame:
        """
        Add row indices to DataFrame.
        Unique indices of type integer will be added in non-consecutive increasing order.

        :return: Spark DataFrame with extra column of row indices.
        """
        return self.__df.withColumn(index_name, monotonically_increasing_id())

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

        :param feature_indices:
        :param keep:
        :return:
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

        :param scaler_method:
        :return:
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
        Split DataFrame into training and test dataset.
        It will generate a nearly class-balanced training
        and testing set for both categorical and continuous label input.

        :param label_type_cat: If True (the default), the input label colum will be processed as categorical.
                               Otherwise, it will be considered a continuous variable and binarized.
        :param split_training_factor: Proportion of the training set. Usually, a value between 0.6 and 0.8.

        :return: Tuple of FSDataFrames.
        """

        row_index_col = self.get_row_index_name()
        label_col = self.get_label_col_name()
        sdf = self.__df

        # create a temporal indexed categorical variable for sampling and splitting the data set.
        tmp_label_col = '_tmp_label_indexed'
        if label_type_cat:
            sdf = _string_indexer(sdf=sdf, input_col=label_col, output_col=tmp_label_col)
        else:
            # If the input label is continuous, create a uniform random distribution [0,1] and binarize this variable.
            # It will be used then as categorical for sampling the dataframe.
            sdf = sdf.withColumn("_tmp_uniform_rand", rand())
            sdf = (_binarizer(sdf,
                              input_col="_tmp_uniform_rand",
                              output_col=tmp_label_col,
                              threshold=0.5,
                              drop_input_col=True)
                   )

        # Get number of levels for categorical variable.
        levels = [lv[tmp_label_col] for lv in sdf.select([tmp_label_col]).distinct().collect()]

        # Sampling DataFrame to extract class-balanced training set.
        # This will keep similar proportion by stratum in both training and testing set.
        fraction_dict = dict(zip(levels, [split_training_factor] * len(levels)))
        training_df = sdf.sampleBy(col=sdf[tmp_label_col], fractions=fraction_dict)

        # Filter out the testing set from the input Dataframe. testing_df = input_sdf[-training_df].
        testing_df = sdf.join(training_df, [row_index_col], "leftanti")

        # Drop tmp cols
        training_df = training_df.drop(tmp_label_col)
        testing_df = testing_df.drop(tmp_label_col)

        return (self.update(training_df, self.__sample_col, self.__label_col, self.__row_index_name),
                self.update(testing_df, self.__sample_col, self.__label_col, self.__row_index_name))

    @classmethod
    def update(cls,
               df: pyspark.sql.DataFrame,
               sample_col: str,
               label_col: str,
               row_index_col: str):
        """
        Create a new instance of FSDataFrame.

        :param df: Spark DataFrame
        :param sample_col: Name of sample id column.
        :param label_col: Name of sample label column.
        :param row_index_col: Name of row (instances) id column.

        :return: FSDataFrame
        """
        return cls(df, sample_col, label_col, row_index_col)


def _assemble_column_vector(sdf: pyspark.sql.DataFrame,
                            input_feature_cols: List[str],
                            output_column_vector: str = 'features',
                            drop_input_cols: bool = True) -> pyspark.sql.DataFrame:
    """
    Assemble features (columns) from DataFrame into a column of type Dense Vector.

    :param drop_input_cols:
    :param sdf: Spark DataFrame
    :param input_feature_cols: List of features column names.
    :param output_column_vector: Output column of type DenseVector.

    :return: Spark DataFrame
    """

    sdf_vector = (VectorAssembler()
                  .setInputCols(input_feature_cols)
                  .setOutputCol(output_column_vector)
                  .transform(sdf)
                  )

    return sdf_vector.drop(*input_feature_cols) if drop_input_cols else sdf_vector


def _disassemble_column_vector(sdf: pyspark.sql.DataFrame,
                               features_cols: List[str],
                               col_vector_name: str,
                               drop_col_vector: bool = True) -> pyspark.sql.DataFrame:
    """
    Convert a Column Dense Vector in Spark DataFrame to individual columns (a.k.a features).
    Basically, revert the operation from `_assemble_column_vector`.

    :param drop_col_vector:
    :param sdf: Spark DataFrame
    :param features_cols:
    :param col_vector_name:

    :return: Spark DataFrame
    """

    sdf = (sdf
           .withColumn("_array_col", vector_to_array(sdf[col_vector_name]))
           .withColumns({features_cols[i]: col("_array_col")[i] for i in range(len(features_cols))})
           .drop("_array_col")
           )

    return sdf.drop(col_vector_name) if drop_col_vector else sdf


def _string_indexer(sdf: pyspark.sql.DataFrame,
                    input_col: str = None,
                    output_col: str = "_label_indexed",
                    drop_input_col: bool = False) -> pyspark.sql.DataFrame:
    """
    Wrapper for `pyspark.ml.feature.StringIndexer`.
    See https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.StringIndexer.html.

    :param sdf:
    :param input_col:
    :param output_col:
    :param drop_input_col:

    :return:
    """
    sdf = (StringIndexer()
           .setInputCol(input_col)
           .setOutputCol(output_col)
           .fit(sdf)
           .transform(sdf)
           )
    return sdf.drop(input_col) if drop_input_col else sdf


def _binarizer(sdf: pyspark.sql.DataFrame,
               input_col: str = None,
               output_col: str = "_label_binarized",
               threshold: float = 0.5,
               drop_input_col: bool = False) -> pyspark.sql.DataFrame:
    """
     Wrapper for `pyspark.ml.feature.Binarizer`.
     See https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Binarizer.html

    :param sdf: Spark DataFrame.
    :param input_col: Name of the numeric input column to be binarized.
    :param output_col: Name of the output column binarized.
    :param threshold: Threshold used to binarize continuous features.
                      The features greater than the threshold will be binarized to 1.0.
                      The features equal to or less than the threshold will be binarized to 0.0
    :param drop_input_col:

    :return:
    """
    sdf = (Binarizer()
           .setInputCol(input_col)
           .setOutputCol(output_col)
           .setThreshold(threshold)
           .transform(sdf)
           )

    return sdf.drop(input_col) if drop_input_col else sdf


class DataFormatError(Exception):
    pass
