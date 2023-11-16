import pytest

def test_prefect_1_import_warning():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ImportError):
        with pytest.warns(UserWarning, match="Attempted import of 'prefect.Client"):
            from prefect import Client