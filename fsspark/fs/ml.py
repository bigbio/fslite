"""

A set of pre-defined ML algorithms wrapped with cross-validation approach
for feature selection (e.g., rank by feature importance) and prediction.

"""

import pyspark.sql
import pyspark.pandas as pd
from pyspark.ml.classification import (
    RandomForestClassifier,
    LinearSVC,
    RandomForestClassificationModel, LinearSVCModel,
)
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    RegressionEvaluator,
    BinaryClassificationEvaluator,
)
from pyspark.ml.regression import RandomForestRegressor, FMRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel

from fsspark.fs.core import FSDataFrame


def cv_rf_classification(
        fsdf: FSDataFrame, binary_classification: bool = True
) -> CrossValidatorModel:
    """
    Cross-validation with Random Forest classifier as estimator.

    :param fsdf: FSDataFrame
    :param binary_classification: If true (default), current problem is considered of type binary classification.
                                  Otherwise, implement a multi-class classification problem.

    :return: CrossValidatorModel
             TODO: Consider here if make sense to return the full CV Model.
    """
    features_col = "features"
    sdf = fsdf.get_sdf_vector(output_column_vector=features_col)
    label_col = fsdf.get_label_col_name()

    # set the estimator (a.k.a. the ML algorithm)
    rf = RandomForestClassifier(
        featuresCol=features_col,
        labelCol=label_col,
        numTrees=3,
        maxDepth=2,
        seed=42,
        leafCol="leafId",
    )
    grid = ParamGridBuilder().addGrid(rf.maxDepth, [2, 3]).build()

    # set the evaluator.
    if binary_classification:
        evaluator = BinaryClassificationEvaluator()
    else:
        evaluator = MulticlassClassificationEvaluator()

    cv = CrossValidator(
        estimator=rf,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        parallelism=2,
        numFolds=3,
        collectSubModels=False,
    )
    cv_model = cv.fit(sdf)
    return cv_model


def cv_svc_classification(
        fsdf: FSDataFrame,
) -> CrossValidatorModel:
    """
    Cross-validation with Linear Support Vector classifier as estimator.
    Support only binary classification.

    :param fsdf: FSDataFrame

    :return: CrossValidatorModel
             TODO: Consider here if make sense to return the full CV Model.
    """

    features_col = "features"
    sdf = fsdf.get_sdf_vector(output_column_vector=features_col)
    label_col = fsdf.get_label_col_name()

    # set the estimator (a.k.a. the ML algorithm)
    svm = LinearSVC(
        featuresCol=features_col, labelCol=label_col, maxIter=10, regParam=0.1
    )

    grid = ParamGridBuilder().addGrid(svm.maxIter, [10]).build()

    # set the evaluator.
    evaluator = BinaryClassificationEvaluator()

    cv = CrossValidator(
        estimator=svm,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        parallelism=2,
        numFolds=3,
        collectSubModels=False,
    )
    cv_model = cv.fit(sdf)
    return cv_model


def cv_rf_regression(fsdf: FSDataFrame) -> CrossValidatorModel:
    """
    Cross-validation with Random Forest regressor as estimator.
    Optimised for regression problems.

    :param fsdf: FSDataFrame

    :return: CrossValidatorModel
             TODO: Consider here if make sense to return the full CV Model.
    """

    features_col = "features"
    sdf = fsdf.get_sdf_vector(output_column_vector=features_col)
    label_col = fsdf.get_label_col_name()

    rf = RandomForestRegressor(
        featuresCol=features_col,
        labelCol=label_col,
        numTrees=3,
        maxDepth=2,
        seed=42,
        leafCol="leafId",
    )
    grid = ParamGridBuilder().addGrid(rf.maxDepth, [2, 3]).build()

    # set the evaluator.
    evaluator = RegressionEvaluator()

    cv = CrossValidator(
        estimator=rf,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        parallelism=2,
        numFolds=3,
        collectSubModels=False,
    )
    cv_model = cv.fit(sdf)
    return cv_model


def cv_fm_regression(fsdf: FSDataFrame) -> CrossValidatorModel:
    """
    Cross-validation with Factorization Machines as estimator.
    Optimised for regression problems.

    :param fsdf: FSDataFrame

    :return: CrossValidatorModel
              TODO: Do it make sense here to return the full CV Model??
    """

    features_col = "features"
    sdf = fsdf.get_sdf_vector(output_column_vector=features_col)
    label_col = fsdf.get_label_col_name()

    fmr = FMRegressor(featuresCol=features_col, labelCol=label_col)

    grid = ParamGridBuilder().addGrid(fmr.factorSize, [4, 8]).build()

    # set the evaluator.
    evaluator = RegressionEvaluator()

    cv = CrossValidator(
        estimator=fmr,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        parallelism=2,
        numFolds=3,
        collectSubModels=False,
    )
    cv_model = cv.fit(sdf)
    return cv_model


def get_accuracy(model: CrossValidatorModel) -> float:
    """
    # TODO: This function should be able to parse all available models.
            Currently only support RandomForestClassificationModel.

    :param model: Trained CrossValidatorModel
    :return: Training accuracy
    """
    best_model = model.bestModel
    if isinstance(best_model, RandomForestClassificationModel):
        acc = best_model.summary.accuracy
    elif isinstance(best_model, LinearSVCModel):
        acc = best_model.summary().accuracy
    else:
        acc = None
    return acc


def get_predictions(model: CrossValidatorModel) -> pyspark.sql.DataFrame:
    """
    # TODO: This function should be able to parse all available models.
            Currently only support RandomForestClassificationModel.

    :param model: Trained CrossValidatorModel

    :return: DataFrame with sample label predictions
    """
    best_model = model.bestModel
    if isinstance(best_model, RandomForestClassificationModel):
        pred = best_model.summary.predictions.drop(
            best_model.getFeaturesCol(),
            best_model.getLeafCol(),
            best_model.getRawPredictionCol(),
            best_model.getProbabilityCol(),
        )
    else:
        pred = None
    return pred


def get_feature_scores(model: CrossValidatorModel,
                       indexed_features: pyspark.pandas.series.Series = None) -> pd.DataFrame:
    """
    Extract features scores (e.g. importance or coefficients) from a trained CrossValidatorModel.

    # TODO: This function should be able to parse all available models.
            Currently only support RandomForestClassificationModel and LinearSVCModel.

    :param model: Trained CrossValidatorModel
    :param indexed_features: If provided, report features names rather than features indices.
                             Usually, the output from `training_data.get_features_indexed()`.

    :return: Pandas on DataFrame with feature importance
    """

    df_features = (None if indexed_features is None
                   else indexed_features.to_dataframe(name='features')
                   )

    best_model = model.bestModel

    if isinstance(best_model, RandomForestClassificationModel):

        importance = pd.DataFrame(data=best_model.featureImportances.toArray(),
                                  columns=['importance'])

        df = (importance
              .reset_index(level=0)
              .rename(columns={"index": "feature_index"})
              )

        if df_features is not None:
            # if available, get feature names rather than reporting feature index.
            df = (df_features
                  .merge(importance, how='right', left_index=True, right_index=True)  # index-to-index merging
                  )

        return df.sort_values(by="importance", ascending=False)

    elif isinstance(best_model, LinearSVCModel):

        coefficients = pd.DataFrame(data=best_model.coefficients,
                                    columns=['coefficients'])

        df = (coefficients
              .reset_index(level=0)
              .rename(columns={"index": "feature_index"})
              )

        if indexed_features is not None:
            df = (df_features
                  .merge(coefficients, how='right', left_index=True, right_index=True)  # index-to-index merging
                  )

        return df.sort_values(by="coefficients", ascending=False)

    else:
        df = None  # this should follow with parsing options for the different models.
        return df
