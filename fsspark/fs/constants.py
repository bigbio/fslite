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

RF_BINARY = 'rf_binary'
RF_MULTILABEL = 'rf_multilabel'
RF_REGRESSION = 'rf_regression'
LSVC_BINARY = 'lsvc_binary'

# Define dict with machine learning wrapper methods and brief description
ML_METHODS = {
    RF_BINARY: 'Random Forest Binary Classifier',
    RF_MULTILABEL: 'Random Forest Multi-label Classifier',
    RF_REGRESSION: 'Random Forest Regression',
    LSVC_BINARY: 'Linear SVC Binary Classifier'
}
