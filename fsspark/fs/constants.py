# Define constants for the project


# Define univariate feature selection methods constants
ANOVA = 'anova'
UNIVARIATE_CORRELATION = 'u_corr'
F_REGRESSION = 'f_regression'

# Define dict with univariate feature selection methods and brief description
UNIVARIATE_METHODS = {
    ANOVA: 'ANOVA univariate feature selection (F-classification)',
    UNIVARIATE_CORRELATION: 'Univariate Correlation',
    F_REGRESSION: 'Univariate F-regression'
}

# Define multivariate feature selection methods constants
MULTIVARIATE_CORRELATION = 'm_corr'
MULTIVARIATE_VARIANCE = 'variance'

# Define dict with multivariate feature selection methods and brief description
MULTIVARIATE_METHODS = {
    MULTIVARIATE_CORRELATION: 'Multivariate Correlation',
    MULTIVARIATE_VARIANCE: 'Multivariate Variance'
}

# Define machine learning wrapper methods constants

# binary classification
RF_BINARY = 'rf_binary'
LSVC_BINARY = 'lsvc_binary'
FM_BINARY = 'fm_binary'  # TODO: implement this method

# multilabel classification
RF_MULTILABEL = 'rf_multilabel'
LR_MULTILABEL = 'lg_multilabel'  # TODO: implement this method

# regression
RF_REGRESSION = 'rf_regression'
FM_REGRESSION = 'fm_regression'  # TODO: implement this method


# Define dict with machine learning wrapper methods and brief description
ML_METHODS = {
    RF_BINARY: 'Random Forest Binary Classifier',
    LSVC_BINARY: 'Linear SVC Binary Classifier',
    FM_BINARY: 'Factorization Machine Binary Classifier',

    RF_MULTILABEL: 'Random Forest Multi-label Classifier',
    LR_MULTILABEL: 'Logistic Regression Multi-label Classifier',

    RF_REGRESSION: 'Random Forest Regression',
    FM_REGRESSION: 'Factorization Machine Regression'
}
