"""
This file contains a list of constants used in the feature selection and machine learning methods.
"""

FS_METHODS = {
    'univariate': {
        "title": 'Univariate Feature Selection',
        "methods": [
            {
                'name': 'anova',
                'description': 'ANOVA univariate feature selection (F-classification)'
            }
        ]
    },
    'multivariate': {
        "title": 'Multivariate Feature Selection',
        "methods": [
            {
                'name': 'm_corr',
                'description': 'Multivariate Correlation'
            },
            {
                'name': 'variance',
                'description': 'Multivariate Variance'
            }
        ]
    },
    'ml': {
        "title": 'Machine Learning Wrapper',
        "methods": [
            {
                'name': 'rf_binary',
                'description': 'Random Forest Binary Classifier'
            },
            {
                'name': 'lsvc_binary',
                'description': 'Linear SVC Binary Classifier'
            },
            {
                'name': 'fm_binary',
                'description': 'Factorization Machine Binary Classifier'
            },
            {
                'name': 'rf_multilabel',
                'description': 'Random Forest Multi-label Classifier'
            },
            {
                'name': 'lg_multilabel',
                'description': 'Logistic Regression Multi-label Classifier'
            },
            {
                'name': 'rf_regression',
                'description': 'Random Forest Regression'
            },
            {
                'name': 'fm_regression',
                'description': 'Factorization Machine Regression'
            }
        ]
    }
}