import logging
from typing import List, Tuple

import numpy
import numpy as np
import pandas as pd
from pandas import DataFrame
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
    ):
        """
        Create an instance of FSDataFrame.

        Expected an input DataFrame with 2+N columns.
        After specifying sample id and sample label columns, the remaining N columns will be considered as features.

        :param df: Pandas DataFrame
        :param sample_col: Sample id column name
        :param label_col: Sample label column name
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
        self.__is_scaled = (False, None)

    def get_feature_matrix(self) -> numpy.array:
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

    def scale_features(self, scaler_method: str = 'standard', **kwargs) -> bool:
        """
        Scales features in the SDataFrame using a specified method.

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

        # TODO: Scale only the features for now, we have to investigate if we scale cateogrical variables
        self.__matrix = scaler.fit_transform(self.__matrix)
        self.__is_scaled = (True, scaler_method)
        return True

    def is_scaled(self):
        return self.__is_scaled[0]

    def get_scaled_method(self):
        return self.__is_scaled[1]

    def select_features_by_index(self, feature_indexes: List[int]) -> 'FSDataFrame':
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
        return FSDataFrame(updated_df, sample_col=self.__sample_col, label_col=self.__label_col)

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

