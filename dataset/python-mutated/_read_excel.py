"""Amazon S3 Excel Read Module (PRIVATE)."""
import logging
from typing import Any, Dict, Optional, Union
import boto3
import awswrangler.pandas as pd
from awswrangler import _utils, exceptions
from awswrangler.s3._fs import open_s3_object
openpyxl = _utils.import_optional_dependency('openpyxl')
_logger: logging.Logger = logging.getLogger(__name__)

@_utils.check_optional_dependency(openpyxl, 'openpyxl')
def read_excel(path: str, version_id: Optional[str]=None, use_threads: Union[bool, int]=True, boto3_session: Optional[boto3.Session]=None, s3_additional_kwargs: Optional[Dict[str, Any]]=None, **pandas_kwargs: Any) -> pd.DataFrame:
    if False:
        return 10
    'Read EXCEL file(s) from a received S3 path.\n\n    Note\n    ----\n    This function accepts any Pandas\'s read_excel() argument.\n    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html\n\n    Note\n    ----\n    Depending on the file extension (\'xlsx\', \'xls\', \'odf\'...), an additional library\n    might have to be installed first.\n\n    Note\n    ----\n    In case of `use_threads=True` the number of threads\n    that will be spawned will be gotten from os.cpu_count().\n\n    Parameters\n    ----------\n    path : str\n        S3 path (e.g. ``s3://bucket/key.xlsx``).\n    version_id : Optional[str]\n        Version id of the object.\n    use_threads : Union[bool, int]\n        True to enable concurrent requests, False to disable multiple threads.\n        If enabled os.cpu_count() will be used as the max number of threads.\n        If given an int will use the given amount of threads.\n        If integer is provided, specified number is used.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n    s3_additional_kwargs : Optional[Dict[str, Any]]\n        Forward to botocore requests, only "SSECustomerAlgorithm" and "SSECustomerKey" arguments will be considered.\n    pandas_kwargs:\n        KEYWORD arguments forwarded to pandas.read_excel(). You can NOT pass `pandas_kwargs` explicit, just add valid\n        Pandas arguments in the function call and awswrangler will accept it.\n        e.g. wr.s3.read_excel("s3://bucket/key.xlsx", na_rep="", verbose=True)\n        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html\n\n    Returns\n    -------\n    pandas.DataFrame\n        Pandas DataFrame.\n\n    Examples\n    --------\n    Reading an EXCEL file\n\n    >>> import awswrangler as wr\n    >>> df = wr.s3.read_excel(\'s3://bucket/key.xlsx\')\n\n    '
    if 'pandas_kwargs' in pandas_kwargs:
        raise exceptions.InvalidArgument("You can NOT pass `pandas_kwargs` explicit, just add valid Pandas arguments in the function call and awswrangler will accept it.e.g. wr.s3.read_excel('s3://bucket/key.xlsx', na_rep='', verbose=True)")
    with open_s3_object(path=path, mode='rb', version_id=version_id, use_threads=use_threads, s3_block_size=-1, s3_additional_kwargs=s3_additional_kwargs, boto3_session=boto3_session) as f:
        _logger.debug('pandas_kwargs: %s', pandas_kwargs)
        return pd.read_excel(f, **pandas_kwargs)