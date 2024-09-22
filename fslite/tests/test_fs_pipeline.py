import unittest

from fslite.config.context import init_spark, stop_spark_session
from fslite.fs.core import FSDataFrame
from fslite.fs.methods import FSPipeline, FSUnivariate, FSMultivariate, FSMLMethod
from fslite.utils.datasets import get_tnbc_data_path
from fslite.utils.io import import_table_as_psdf


class FeatureSelectionPipelineTest(unittest.TestCase):

    def setUp(self) -> None:
        init_spark(
            apply_pyarrow_settings=True,
            apply_extra_spark_settings=True,
            apply_pandas_settings=True,
        )

    def tearDown(self) -> None:
        stop_spark_session()

    @staticmethod
    def import_FSDataFrame():
        df = import_table_as_psdf(get_tnbc_data_path(), n_partitions=5)
        fsdf = FSDataFrame(df, sample_col="Sample", label_col="label")
        return fsdf

    def test_feature_selection_pipeline(self):
        fsdf = self.import_FSDataFrame()

        training_data, testing_data = fsdf.split_df(split_training_factor=0.6)

        # create a Univariate object
        univariate = FSUnivariate(
            fs_method="anova", selection_mode="percentile", selection_threshold=0.8
        )

        # create a Multivariate object
        multivariate = FSMultivariate(
            fs_method="m_corr", corr_threshold=0.75, corr_method="pearson"
        )

        # create a MLMethod object
        rf_classifier = FSMLMethod(
            fs_method="rf_multilabel",
            rfe=True,
            rfe_iterations=2,
            percent_to_keep=0.9,
            estimator_params={"labelCol": "label"},
            evaluator_params={"metricName": "accuracy"},
            grid_params={"numTrees": [10, 15], "maxDepth": [5, 10]},
            cv_params={"parallelism": 2, "numFolds": 5},
        )

        # create a pipeline object
        fs_pipeline = FSPipeline(
            df_training=training_data,
            df_testing=testing_data,
            fs_stages=[univariate, multivariate, rf_classifier],
        )

        # run the pipeline
        results = fs_pipeline.run()

        # print results
        print(results)

        assert results.get("training_metric") > 0.9


if __name__ == "__main__":
    unittest.main()
