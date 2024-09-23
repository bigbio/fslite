"""
This file contains a list of constants used in the feature selection and machine learning methods.
"""

from typing import Dict, List, Union

FS_METHODS = {
    "univariate": {
        "title": "Univariate Feature Selection",
        "methods": [
            {
                "name": "anova",
                "description": "Univariate ANOVA feature selection (f-classification)",
            },
            {"name": "u_corr", "description": "Univariate correlation"},
            {"name": "f_regression", "description": "Univariate f-regression"},
        ],
    },
    "multivariate": {
        "title": "Multivariate Feature Selection",
        "methods": [
            {"name": "m_corr", "description": "Multivariate Correlation"},
            {"name": "variance", "description": "Multivariate Variance"},
        ],
    },
    "ml": {
        "title": "Machine Learning Wrapper",
        "methods": [
            {"name": "rf_binary", "description": "Random Forest Binary Classifier"},
            {"name": "lsvc_binary", "description": "Linear SVC Binary Classifier"},
            {
                "name": "fm_binary",
                "description": "Factorization Machine Binary Classifier",
            },
            {
                "name": "rf_multilabel",
                "description": "Random Forest Multi-label Classifier",
            },
            {
                "name": "lg_multilabel",
                "description": "Logistic Regression Multi-label Classifier",
            },
            {"name": "rf_regression", "description": "Random Forest Regression"},
            {
                "name": "fm_regression",
                "description": "Factorization Machine Regression",
            },
        ],
    },
}


def get_fs_methods():
    """
    Get the list of feature selection methods
    :return: dict
    """
    return FS_METHODS


def get_fs_method_details(method_name: str) -> Union[Dict, None]:
    """
    Get the details of the feature selection method, this function search in all-methods definitions
    and get the details of the method with the given name. If the method is not found, it returns None.
    The method name is case-insensitive.
    :param method_name: str
    :return: dict
    """

    for method_type in FS_METHODS:
        for method in FS_METHODS[method_type]["methods"]:
            if method["name"].lower() == method_name.lower():
                return method
    return None


def get_fs_univariate_methods() -> List:
    """
    Get the list of univariate methods implemented in the library
    :return: list
    """
    return get_fs_method_by_class("univariate")


def get_fs_multivariate_methods() -> List:
    return get_fs_method_by_class("multivariate")


def get_fs_ml_methods() -> List:
    return get_fs_method_by_class("ml")


def is_valid_univariate_method(method_name: str) -> bool:
    """
    This method check if the given method name is a supported univariate method
    :param method_name method name
    :return: boolean
    """
    for method in FS_METHODS["univariate"]["methods"]:
        if method["name"].lower() == method_name:
            return True
    return False


def is_valid_multivariate_method(method_name: str) -> bool:
    """
    This method check if the given method name is a supported multivariate method
    :param method_name method name
    :return: boolean
    """
    for method in FS_METHODS["multivariate"]["methods"]:
        if method["name"].lower() == method_name:
            return True
    return False


def is_valid_ml_method(method_name: str) -> bool:
    """
    This method check if the given method name is a supported machine learning method
    :param method_name method name
    :return: boolean
    """
    for method in FS_METHODS["ml"]["methods"]:
        if method["name"].lower() == method_name:
            return True
    return False


def get_fs_method_by_class(fs_class: str) -> List:
    """
    Get the FS method supported for a given FS class, for example, univariate
    :param fs_class
    :return FS List
    """
    fs_methods = FS_METHODS[fs_class]
    fs_names = [method["name"] for method in fs_methods["methods"]]
    return fs_names
