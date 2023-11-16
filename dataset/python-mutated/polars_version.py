try:
    from polars.polars import get_polars_version as _get_polars_version
    polars_version_string = _get_polars_version()
except ImportError:
    import warnings
    warnings.warn('polars binary missing!', stacklevel=2)
    polars_version_string = ''

def get_polars_version() -> str:
    if False:
        return 10
    '\n    Return the version of the Python Polars package as a string.\n\n    If the Polars binary is missing, returns an empty string.\n    '
    return polars_version_string