"""
This file contains a list of constants used in the feature selection and machine learning methods.
"""

from typing import Dict, List, Union

FS_METHODS = {
    "univariate": {
        "title": "Univariate Feature Selection",
        "description": "Univariate feature selection refers to the process of selecting the most relevant features for "
                       "a machine learning model by evaluating each feature individually with respect to the target "
                       "variable using univariate statistical tests. It simplifies the feature selection process by "
                       "treating each feature independently and assessing its contribution to the predictive "
                       "performance of the model.",
        "methods": [
            {"name": "anova", "description": "Univariate ANOVA feature selection (f-classification)"},
            {"name": "u_corr", "description": "Univariate Pearson's correlation"},
            {"name": "f_regression", "description": "Univariate f-regression"},
            {"name": "mutual_info_regression", "description": "Univariate mutual information regression"},
            {"name": "mutual_info_classification", "description": "Univariate mutual information classification"},
        ],
    },
    "multivariate": {
        "title": "Multivariate Feature Selection",
        "description": "Multivariate feature selection is a method of selecting features by evaluating them in "
                       "combination rather than individually. Unlike univariate feature selection, which treats each "
                       "feature separately, multivariate feature selection considers the relationships and interactions "
                       "between multiple features and the target variable. This method aims to identify a subset of "
                       "features that work well together to improve the performance of a machine learning model.",
        "methods": [
            {"name": "m_corr", "description": "Multivariate Correlation"},
            {"name": "variance", "description": "Multivariate Variance"},
        ],
    },
    "ml": {
        "title": "Machine Learning Wrapper",
        "description": "Machine learning wrapper methods are feature selection techniques that use a machine learning ",
        "methods": [
            {"name": "rf_binary", "description": "Random Forest Binary Classifier"},
            {"name": "lsvc_binary", "description": "Linear SVC Binary Classifier"},
            {"name": "fm_binary", "description": "Factorization Machine Binary Classifier"},
            {"name": "rf_multilabel", "description": "Random Forest Multi-label Classifier"},
            {"name": "lg_multilabel","description": "Logistic Regression Multi-label Classifier"},
            {"name": "rf_regression", "description": "Random Forest Regression"},
            {"name": "fm_regression","description": "Factorization Machine Regression"},
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
