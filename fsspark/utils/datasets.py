from pathlib import Path


ROOT_PATH = Path(__file__).parent.parent


def get_tnbc_data_path() -> str:
    """
    Return path to example dataset (TNBC) with 44 samples and 500 features.

    """
    tnbc_path = Path(__file__).parent.parent / "testdata/TNBC.tsv.gz"
    return tnbc_path.__str__()


def get_tnbc_missing_path() -> str:
    """
    Return path to example dataset (TNBC) with missing values.

    """
    tnbc_path = Path(__file__).parent.parent / "testdata/TNBC_missing.tsv"
    return tnbc_path.__str__()
