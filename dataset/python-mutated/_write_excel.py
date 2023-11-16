"""Amazon S3 Excel Write Module (PRIVATE)."""
import logging
from typing import Any, Dict, Optional, Union
import boto3
import pandas as pd
from awswrangler import _utils, exceptions
from awswrangler.s3._fs import open_s3_object
openpyxl = _utils.import_optional_dependency('openpyxl')
_logger: logging.Logger = logging.getLogger(__name__)

@_utils.check_optional_dependency(openpyxl, 'openpyxl')
def to_excel(df: pd.DataFrame, path: str, boto3_session: Optional[boto3.Session]=None, s3_additional_kwargs: Optional[Dict[str, Any]]=None, use_threads: Union[bool, int]=True, **pandas_kwargs: Any) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Write EXCEL file on Amazon S3.\n\n    Note\n    ----\n    This function accepts any Pandas\'s read_excel() argument.\n    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html\n\n    Note\n    ----\n    Depending on the file extension (\'xlsx\', \'xls\', \'odf\'...), an additional library\n    might have to be installed first.\n\n    Note\n    ----\n    In case of `use_threads=True` the number of threads\n    that will be spawned will be gotten from os.cpu_count().\n\n    Parameters\n    ----------\n    df: pandas.DataFrame\n        Pandas DataFrame https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html\n    path : str\n        Amazon S3 path (e.g. s3://bucket/filename.xlsx).\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 Session will be used if boto3_session receive None.\n    s3_additional_kwargs : Optional[Dict[str, Any]]\n        Forwarded to botocore requests.\n        e.g. s3_additional_kwargs={\'ServerSideEncryption\': \'aws:kms\', \'SSEKMSKeyId\': \'YOUR_KMS_KEY_ARN\'}\n    use_threads : bool, int\n        True to enable concurrent requests, False to disable multiple threads.\n        If enabled os.cpu_count() will be used as the max number of threads.\n        If integer is provided, specified number is used.\n    pandas_kwargs:\n        KEYWORD arguments forwarded to pandas.DataFrame.to_excel(). You can NOT pass `pandas_kwargs` explicit, just add\n        valid Pandas arguments in the function call and awswrangler will accept it.\n        e.g. wr.s3.to_excel(df, path, na_rep="", index=False)\n        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_excel.html\n\n    Returns\n    -------\n    str\n        Written S3 path.\n\n    Examples\n    --------\n    Writing EXCEL file\n\n    >>> import awswrangler as wr\n    >>> import pandas as pd\n    >>> wr.s3.to_excel(df, \'s3://bucket/filename.xlsx\')\n\n    '
    if 'pandas_kwargs' in pandas_kwargs:
        raise exceptions.InvalidArgument('You can NOT pass `pandas_kwargs` explicit, just add valid Pandas arguments in the function call and awswrangler will accept it.e.g. wr.s3.to_excel(df, path, na_rep=, index=False)')
    with open_s3_object(path=path, mode='wb', use_threads=use_threads, s3_additional_kwargs=s3_additional_kwargs, boto3_session=boto3_session) as f:
        _logger.debug('pandas_kwargs: %s', pandas_kwargs)
        df.to_excel(f, **pandas_kwargs)
    return path