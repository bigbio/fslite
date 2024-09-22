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


def get_fs_methods():
    """
    Get the list of feature selection methods
    :return: dict
    """
    return FS_METHODS

def get_fs_method_details(method_name: str):
    """
    Get the details of the feature selection method, this function search in all-methods definitions
    and get the details of the method with the given name. If the method is not found, it returns None.
    The method name is case-insensitive.
    :param method_name: str
    :return: dict
    """

    for method_type in FS_METHODS:
        for method in FS_METHODS[method_type]['methods']:
            if method['name'].lower() == method_name.lower():
                return method
    return None
