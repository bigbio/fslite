"""
A set of generic functions that are used in the project.
"""


# Generic decorator to label a function with a specified tag.
def tag(label: str):
    """
    Decorator to tag a function with a specified tag (e.g., experimental).

    :param label: tag label
    :return: decorator
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"Tag for {func.__name__}: {label}")
            return func(*args, **kwargs)

        return wrapper
    return decorator
