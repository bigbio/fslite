import unittest

from pyspark.ml.classification import (RandomForestClassifier,
                                       LogisticRegression)
from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                   MulticlassClassificationEvaluator)

from fsspark.config.context import init_spark, stop_spark_session
from fsspark.fs.core import FSDataFrame
from fsspark.fs.ml import MLCVModel
from fsspark.utils.datasets import get_tnbc_data_path
from fsspark.utils.io import import_table_as_psdf


class MLMethodTest(unittest.TestCase):

    def setUp(self) -> None:
        init_spark(apply_pyarrow_settings=True,
                   apply_extra_spark_settings=True,
                   apply_pandas_settings=True)

    def tearDown(self) -> None:
        stop_spark_session()

    @staticmethod
    def import_FSDataFrame():
        df = import_table_as_psdf(get_tnbc_data_path(), n_partitions=5)
        fsdf = FSDataFrame(df, sample_col='Sample', label_col='label')
        return fsdf

    def test_build_model_using_cross_validator(self):
        fsdf = self.import_FSDataFrame()
        estimator = RandomForestClassifier()
        evaluator = BinaryClassificationEvaluator()
        grid_params = {'numTrees': [10, 20, 30], 'maxDepth': [5, 10, 15]}
        ml_method = MLCVModel(
            estimator=estimator,
            evaluator=evaluator,
            estimator_params=None,
            grid_params=None,
            cv_params=None
        )

        print(ml_method._cross_validator.__str__())
        assert ml_method._cross_validator is not None

    def test_get_feature_scores_random_forest_classifier(self):
        # Create a sample FSDataFrame
        fsdf = self.import_FSDataFrame()

        # Create a RandomForestClassifier model
        estimator = RandomForestClassifier()
        evaluator = MulticlassClassificationEvaluator()
        estimator_params = {'labelCol': 'label'}
        grid_params = {'numTrees': [10, 20, 30], 'maxDepth': [5, 10, 15]}
        cv_params = {'parallelism': 2, 'numFolds': 5, 'collectSubModels': False}

        ml_method = MLCVModel(
            estimator=estimator,
            evaluator=evaluator,
            estimator_params=estimator_params,
            grid_params=grid_params,
            cv_params=cv_params
        )

        (ml_method
         .fit(fsdf)
         )

        # Get the feature scores
        feature_scores = ml_method.get_feature_scores()

        # Assert that the feature scores DataFrame is not empty
        assert not feature_scores.empty

        # Assert that the feature scores DataFrame has the expected columns
        expected_columns = ['features', 'feature_index', 'scores', 'percentile_rank']
        assert list(feature_scores.columns) == expected_columns

        # check if dataframe is sorted by scores (descending)
        assert feature_scores['scores'].is_monotonic_decreasing

        print(feature_scores)

    def test_multilabel_rf_model(self):
        fsdf = self.import_FSDataFrame()
        training_data, testing_data = fsdf.split_df(split_training_factor=0.8)

        estimator = RandomForestClassifier()
        evaluator = MulticlassClassificationEvaluator(metricName='accuracy')
        estimator_params = {'labelCol': 'label'}
        grid_params = {'numTrees': [5, 10], 'maxDepth': [3, 5]}
        cv_params = {'parallelism': 2, 'numFolds': 3}

        ml_method = MLCVModel(
            estimator=estimator,
            evaluator=evaluator,
            estimator_params=estimator_params,
            grid_params=grid_params,
            cv_params=cv_params
        )

        (ml_method
         .fit(training_data)
         )

        # get the accuracy on training
        eval_training = ml_method.get_eval_metric_on_training()
        print(f"Accuracy on training data: {eval_training}")

        # get the accuracy on testing
        testing_acc = ml_method.get_eval_metric_on_testing(testing_data)
        print(f"Accuracy on test data: {testing_acc}")
        assert testing_acc > 0.7

    def test_multilabel_lr_model(self):
        fsdf = self.import_FSDataFrame()
        training_data, testing_data = fsdf.split_df(split_training_factor=0.6)

        estimator = LogisticRegression()
        evaluator = MulticlassClassificationEvaluator(metricName='accuracy')
        estimator_params = {'labelCol': 'label'}
        grid_params = {'regParam': [0.1, 0.01]}
        cv_params = {'parallelism': 2, 'numFolds': 3}

        ml_method = MLCVModel(
            estimator=estimator,
            evaluator=evaluator,
            estimator_params=estimator_params,
            grid_params=grid_params,
            cv_params=cv_params
        )

        (ml_method
         .fit(training_data)
         )

        # get the accuracy on training
        eval_training = ml_method.get_eval_metric_on_training()
        print(f"Accuracy on training data: {eval_training}")

        # get the accuracy on testing
        testing_acc = ml_method.get_eval_metric_on_testing(testing_data)
        print(f"Accuracy on test data: {testing_acc}")
        assert testing_acc > 0.7

    def test_FSMLMethod(self):
        from fsspark.fs.methods import FSMLMethod

        fsdf = self.import_FSDataFrame()
        training_data, testing_data = fsdf.split_df(split_training_factor=0.7)

        estimator_params = {'labelCol': 'label'}
        grid_params = {'numTrees': [5, 10], 'maxDepth': [3, 5]}
        cv_params = {'parallelism': 2, 'numFolds': 3}

        ml_method = FSMLMethod(
            fs_method='rf_multilabel',
            rfe=True,
            rfe_iterations=2,
            percent_to_keep=0.9,
            estimator_params=estimator_params,
            evaluator_params={'metricName': 'accuracy'},
            grid_params=grid_params,
            cv_params=cv_params
        )

        filtered_fsdf = ml_method.select_features(training_data)

        training_acc = ml_method.get_eval_metric_on_training()
        print(f"Training accuracy: {training_acc}")
        assert training_acc > 0.8

        testing_acc = ml_method.get_eval_metric_on_testing(testing_data)
        print(f"Testing accuracy: {testing_acc}")
        assert testing_acc > 0.7


if __name__ == '__main__':
    unittest.main()
