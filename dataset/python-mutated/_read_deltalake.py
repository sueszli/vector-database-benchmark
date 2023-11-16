"""Amazon S3 Read Delta Lake Module (PRIVATE)."""
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import boto3
from typing_extensions import Literal
import awswrangler.pandas as pd
from awswrangler import _data_types, _utils
from awswrangler._config import apply_configs
if TYPE_CHECKING:
    try:
        import deltalake
    except ImportError:
        pass
else:
    deltalake = _utils.import_optional_dependency('deltalake')

def _set_default_storage_options_kwargs(boto3_session: Optional[boto3.Session], s3_additional_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if False:
        i = 10
        return i + 15
    defaults = {key.upper(): value for (key, value) in _utils.boto3_to_primitives(boto3_session=boto3_session).items()}
    defaults['AWS_REGION'] = defaults.pop('REGION_NAME')
    s3_additional_kwargs = s3_additional_kwargs or {}
    return {**defaults, **s3_additional_kwargs}

@_utils.check_optional_dependency(deltalake, 'deltalake')
@apply_configs
def read_deltalake(path: str, version: Optional[int]=None, partitions: Optional[List[Tuple[str, str, Any]]]=None, columns: Optional[List[str]]=None, without_files: bool=False, dtype_backend: Literal['numpy_nullable', 'pyarrow']='numpy_nullable', use_threads: bool=True, boto3_session: Optional[boto3.Session]=None, s3_additional_kwargs: Optional[Dict[str, str]]=None, pyarrow_additional_kwargs: Optional[Dict[str, Any]]=None) -> pd.DataFrame:
    if False:
        print('Hello World!')
    'Load a Deltalake table data from an S3 path.\n\n    This function requires the `deltalake package\n    <https://delta-io.github.io/delta-rs/python>`__.\n    See the `How to load a Delta table\n    <https://delta-io.github.io/delta-rs/python/usage.html#loading-a-delta-table>`__\n    guide for loading instructions.\n\n    Parameters\n    ----------\n    path: str\n        The path of the DeltaTable.\n    version: Optional[int]\n        The version of the DeltaTable.\n    partitions: Optional[List[Tuple[str, str, Any]]\n        A list of partition filters, see help(DeltaTable.files_by_partitions)\n        for filter syntax.\n    columns: Optional[List[str]]\n        The columns to project. This can be a list of column names to include\n        (order and duplicates are preserved).\n    without_files: bool\n        If True, load the table without tracking files (memory-friendly).\n        Some append-only applications might not need to track files.\n    dtype_backend: str, optional\n        Which dtype_backend to use, e.g. whether a DataFrame should have NumPy arrays,\n        nullable dtypes are used for all dtypes that have a nullable implementation when\n        “numpy_nullable” is set, pyarrow is used for all dtypes if “pyarrow” is set.\n\n        The dtype_backends are still experimential. The "pyarrow" backend is only supported with Pandas 2.0 or above.\n    use_threads : bool\n        True to enable concurrent requests, False to disable multiple threads.\n        When enabled, os.cpu_count() is used as the max number of threads.\n    boto3_session: Optional[boto3.Session()]\n        Boto3 Session. If None, the default boto3 session is used.\n    s3_additional_kwargs: Optional[Dict[str, str]]\n        Forwarded to the Delta Table class for the storage options of the S3 backend.\n    pyarrow_additional_kwargs: Optional[Dict[str, str]]\n        Forwarded to the PyArrow to_pandas method.\n\n    Returns\n    -------\n    df: pd.DataFrame\n        DataFrame with the results.\n\n    See Also\n    --------\n    deltalake.DeltaTable : Create a DeltaTable instance with the deltalake library.\n    '
    arrow_kwargs = _data_types.pyarrow2pandas_defaults(use_threads=use_threads, kwargs=pyarrow_additional_kwargs, dtype_backend=dtype_backend)
    storage_options = _set_default_storage_options_kwargs(boto3_session, s3_additional_kwargs)
    return deltalake.DeltaTable(table_uri=path, version=version, storage_options=storage_options, without_files=without_files).to_pyarrow_table(partitions=partitions, columns=columns).to_pandas(**arrow_kwargs)