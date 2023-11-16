import gzip
import hashlib
import json
import os
import shutil
import time
from contextlib import closing
from functools import wraps
from os.path import join
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from warnings import warn
import numpy as np
from ..utils import Bunch, check_pandas_support
from ..utils._param_validation import Hidden, Integral, Interval, Real, StrOptions, validate_params
from . import get_data_home
from ._arff_parser import load_arff_from_gzip_file
__all__ = ['fetch_openml']
_OPENML_PREFIX = 'https://api.openml.org/'
_SEARCH_NAME = 'api/v1/json/data/list/data_name/{}/limit/2'
_DATA_INFO = 'api/v1/json/data/{}'
_DATA_FEATURES = 'api/v1/json/data/features/{}'
_DATA_QUALITIES = 'api/v1/json/data/qualities/{}'
_DATA_FILE = 'data/v1/download/{}'
OpenmlQualitiesType = List[Dict[str, str]]
OpenmlFeaturesType = List[Dict[str, str]]

def _get_local_path(openml_path: str, data_home: str) -> str:
    if False:
        while True:
            i = 10
    return os.path.join(data_home, 'openml.org', openml_path + '.gz')

def _retry_with_clean_cache(openml_path: str, data_home: Optional[str], no_retry_exception: Optional[Exception]=None) -> Callable:
    if False:
        print('Hello World!')
    'If the first call to the decorated function fails, the local cached\n    file is removed, and the function is called again. If ``data_home`` is\n    ``None``, then the function is called once. We can provide a specific\n    exception to not retry on using `no_retry_exception` parameter.\n    '

    def decorator(f):
        if False:
            return 10

        @wraps(f)
        def wrapper(*args, **kw):
            if False:
                i = 10
                return i + 15
            if data_home is None:
                return f(*args, **kw)
            try:
                return f(*args, **kw)
            except URLError:
                raise
            except Exception as exc:
                if no_retry_exception is not None and isinstance(exc, no_retry_exception):
                    raise
                warn('Invalid cache, redownloading file', RuntimeWarning)
                local_path = _get_local_path(openml_path, data_home)
                if os.path.exists(local_path):
                    os.unlink(local_path)
                return f(*args, **kw)
        return wrapper
    return decorator

def _retry_on_network_error(n_retries: int=3, delay: float=1.0, url: str='') -> Callable:
    if False:
        return 10
    "If the function call results in a network error, call the function again\n    up to ``n_retries`` times with a ``delay`` between each call. If the error\n    has a 412 status code, don't call the function again as this is a specific\n    OpenML error.\n    The url parameter is used to give more information to the user about the\n    error.\n    "

    def decorator(f):
        if False:
            i = 10
            return i + 15

        @wraps(f)
        def wrapper(*args, **kwargs):
            if False:
                print('Hello World!')
            retry_counter = n_retries
            while True:
                try:
                    return f(*args, **kwargs)
                except (URLError, TimeoutError) as e:
                    if isinstance(e, HTTPError) and e.code == 412:
                        raise
                    if retry_counter == 0:
                        raise
                    warn(f'A network error occurred while downloading {url}. Retrying...')
                    retry_counter -= 1
                    time.sleep(delay)
        return wrapper
    return decorator

def _open_openml_url(openml_path: str, data_home: Optional[str], n_retries: int=3, delay: float=1.0):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns a resource from OpenML.org. Caches it to data_home if required.\n\n    Parameters\n    ----------\n    openml_path : str\n        OpenML URL that will be accessed. This will be prefixes with\n        _OPENML_PREFIX.\n\n    data_home : str\n        Directory to which the files will be cached. If None, no caching will\n        be applied.\n\n    n_retries : int, default=3\n        Number of retries when HTTP errors are encountered. Error with status\n        code 412 won't be retried as they represent OpenML generic errors.\n\n    delay : float, default=1.0\n        Number of seconds between retries.\n\n    Returns\n    -------\n    result : stream\n        A stream to the OpenML resource.\n    "

    def is_gzip_encoded(_fsrc):
        if False:
            for i in range(10):
                print('nop')
        return _fsrc.info().get('Content-Encoding', '') == 'gzip'
    req = Request(_OPENML_PREFIX + openml_path)
    req.add_header('Accept-encoding', 'gzip')
    if data_home is None:
        fsrc = _retry_on_network_error(n_retries, delay, req.full_url)(urlopen)(req)
        if is_gzip_encoded(fsrc):
            return gzip.GzipFile(fileobj=fsrc, mode='rb')
        return fsrc
    local_path = _get_local_path(openml_path, data_home)
    (dir_name, file_name) = os.path.split(local_path)
    if not os.path.exists(local_path):
        os.makedirs(dir_name, exist_ok=True)
        try:
            with TemporaryDirectory(dir=dir_name) as tmpdir:
                with closing(_retry_on_network_error(n_retries, delay, req.full_url)(urlopen)(req)) as fsrc:
                    opener: Callable
                    if is_gzip_encoded(fsrc):
                        opener = open
                    else:
                        opener = gzip.GzipFile
                    with opener(os.path.join(tmpdir, file_name), 'wb') as fdst:
                        shutil.copyfileobj(fsrc, fdst)
                shutil.move(fdst.name, local_path)
        except Exception:
            if os.path.exists(local_path):
                os.unlink(local_path)
            raise
    return gzip.GzipFile(local_path, 'rb')

class OpenMLError(ValueError):
    """HTTP 412 is a specific OpenML error code, indicating a generic error"""
    pass

def _get_json_content_from_openml_api(url: str, error_message: Optional[str], data_home: Optional[str], n_retries: int=3, delay: float=1.0) -> Dict:
    if False:
        print('Hello World!')
    "\n    Loads json data from the openml api.\n\n    Parameters\n    ----------\n    url : str\n        The URL to load from. Should be an official OpenML endpoint.\n\n    error_message : str or None\n        The error message to raise if an acceptable OpenML error is thrown\n        (acceptable error is, e.g., data id not found. Other errors, like 404's\n        will throw the native error message).\n\n    data_home : str or None\n        Location to cache the response. None if no cache is required.\n\n    n_retries : int, default=3\n        Number of retries when HTTP errors are encountered. Error with status\n        code 412 won't be retried as they represent OpenML generic errors.\n\n    delay : float, default=1.0\n        Number of seconds between retries.\n\n    Returns\n    -------\n    json_data : json\n        the json result from the OpenML server if the call was successful.\n        An exception otherwise.\n    "

    @_retry_with_clean_cache(url, data_home=data_home)
    def _load_json():
        if False:
            return 10
        with closing(_open_openml_url(url, data_home, n_retries=n_retries, delay=delay)) as response:
            return json.loads(response.read().decode('utf-8'))
    try:
        return _load_json()
    except HTTPError as error:
        if error.code != 412:
            raise error
    raise OpenMLError(error_message)

def _get_data_info_by_name(name: str, version: Union[int, str], data_home: Optional[str], n_retries: int=3, delay: float=1.0):
    if False:
        for i in range(10):
            print('nop')
    '\n    Utilizes the openml dataset listing api to find a dataset by\n    name/version\n    OpenML api function:\n    https://www.openml.org/api_docs#!/data/get_data_list_data_name_data_name\n\n    Parameters\n    ----------\n    name : str\n        name of the dataset\n\n    version : int or str\n        If version is an integer, the exact name/version will be obtained from\n        OpenML. If version is a string (value: "active") it will take the first\n        version from OpenML that is annotated as active. Any other string\n        values except "active" are treated as integer.\n\n    data_home : str or None\n        Location to cache the response. None if no cache is required.\n\n    n_retries : int, default=3\n        Number of retries when HTTP errors are encountered. Error with status\n        code 412 won\'t be retried as they represent OpenML generic errors.\n\n    delay : float, default=1.0\n        Number of seconds between retries.\n\n    Returns\n    -------\n    first_dataset : json\n        json representation of the first dataset object that adhired to the\n        search criteria\n\n    '
    if version == 'active':
        url = _SEARCH_NAME.format(name) + '/status/active/'
        error_msg = 'No active dataset {} found.'.format(name)
        json_data = _get_json_content_from_openml_api(url, error_msg, data_home=data_home, n_retries=n_retries, delay=delay)
        res = json_data['data']['dataset']
        if len(res) > 1:
            warn('Multiple active versions of the dataset matching the name {name} exist. Versions may be fundamentally different, returning version {version}.'.format(name=name, version=res[0]['version']))
        return res[0]
    url = (_SEARCH_NAME + '/data_version/{}').format(name, version)
    try:
        json_data = _get_json_content_from_openml_api(url, error_message=None, data_home=data_home, n_retries=n_retries, delay=delay)
    except OpenMLError:
        url += '/status/deactivated'
        error_msg = 'Dataset {} with version {} not found.'.format(name, version)
        json_data = _get_json_content_from_openml_api(url, error_msg, data_home=data_home, n_retries=n_retries, delay=delay)
    return json_data['data']['dataset'][0]

def _get_data_description_by_id(data_id: int, data_home: Optional[str], n_retries: int=3, delay: float=1.0) -> Dict[str, Any]:
    if False:
        return 10
    url = _DATA_INFO.format(data_id)
    error_message = 'Dataset with data_id {} not found.'.format(data_id)
    json_data = _get_json_content_from_openml_api(url, error_message, data_home=data_home, n_retries=n_retries, delay=delay)
    return json_data['data_set_description']

def _get_data_features(data_id: int, data_home: Optional[str], n_retries: int=3, delay: float=1.0) -> OpenmlFeaturesType:
    if False:
        print('Hello World!')
    url = _DATA_FEATURES.format(data_id)
    error_message = 'Dataset with data_id {} not found.'.format(data_id)
    json_data = _get_json_content_from_openml_api(url, error_message, data_home=data_home, n_retries=n_retries, delay=delay)
    return json_data['data_features']['feature']

def _get_data_qualities(data_id: int, data_home: Optional[str], n_retries: int=3, delay: float=1.0) -> OpenmlQualitiesType:
    if False:
        return 10
    url = _DATA_QUALITIES.format(data_id)
    error_message = 'Dataset with data_id {} not found.'.format(data_id)
    json_data = _get_json_content_from_openml_api(url, error_message, data_home=data_home, n_retries=n_retries, delay=delay)
    return json_data.get('data_qualities', {}).get('quality', [])

def _get_num_samples(data_qualities: OpenmlQualitiesType) -> int:
    if False:
        while True:
            i = 10
    'Get the number of samples from data qualities.\n\n    Parameters\n    ----------\n    data_qualities : list of dict\n        Used to retrieve the number of instances (samples) in the dataset.\n\n    Returns\n    -------\n    n_samples : int\n        The number of samples in the dataset or -1 if data qualities are\n        unavailable.\n    '
    default_n_samples = -1
    qualities = {d['name']: d['value'] for d in data_qualities}
    return int(float(qualities.get('NumberOfInstances', default_n_samples)))

def _load_arff_response(url: str, data_home: Optional[str], parser: str, output_type: str, openml_columns_info: dict, feature_names_to_select: List[str], target_names_to_select: List[str], shape: Optional[Tuple[int, int]], md5_checksum: str, n_retries: int=3, delay: float=1.0, read_csv_kwargs: Optional[Dict]=None):
    if False:
        print('Hello World!')
    'Load the ARFF data associated with the OpenML URL.\n\n    In addition of loading the data, this function will also check the\n    integrity of the downloaded file from OpenML using MD5 checksum.\n\n    Parameters\n    ----------\n    url : str\n        The URL of the ARFF file on OpenML.\n\n    data_home : str\n        The location where to cache the data.\n\n    parser : {"liac-arff", "pandas"}\n        The parser used to parse the ARFF file.\n\n    output_type : {"numpy", "pandas", "sparse"}\n        The type of the arrays that will be returned. The possibilities are:\n\n        - `"numpy"`: both `X` and `y` will be NumPy arrays;\n        - `"sparse"`: `X` will be sparse matrix and `y` will be a NumPy array;\n        - `"pandas"`: `X` will be a pandas DataFrame and `y` will be either a\n          pandas Series or DataFrame.\n\n    openml_columns_info : dict\n        The information provided by OpenML regarding the columns of the ARFF\n        file.\n\n    feature_names_to_select : list of str\n        The list of the features to be selected.\n\n    target_names_to_select : list of str\n        The list of the target variables to be selected.\n\n    shape : tuple or None\n        With `parser="liac-arff"`, when using a generator to load the data,\n        one needs to provide the shape of the data beforehand.\n\n    md5_checksum : str\n        The MD5 checksum provided by OpenML to check the data integrity.\n\n    n_retries : int, default=3\n        The number of times to retry downloading the data if it fails.\n\n    delay : float, default=1.0\n        The delay between two consecutive downloads in seconds.\n\n    read_csv_kwargs : dict, default=None\n        Keyword arguments to pass to `pandas.read_csv` when using the pandas parser.\n        It allows to overwrite the default options.\n\n        .. versionadded:: 1.3\n\n    Returns\n    -------\n    X : {ndarray, sparse matrix, dataframe}\n        The data matrix.\n\n    y : {ndarray, dataframe, series}\n        The target.\n\n    frame : dataframe or None\n        A dataframe containing both `X` and `y`. `None` if\n        `output_array_type != "pandas"`.\n\n    categories : list of str or None\n        The names of the features that are categorical. `None` if\n        `output_array_type == "pandas"`.\n    '
    gzip_file = _open_openml_url(url, data_home, n_retries=n_retries, delay=delay)
    with closing(gzip_file):
        md5 = hashlib.md5()
        for chunk in iter(lambda : gzip_file.read(4096), b''):
            md5.update(chunk)
        actual_md5_checksum = md5.hexdigest()
    if actual_md5_checksum != md5_checksum:
        raise ValueError(f'md5 checksum of local file for {url} does not match description: expected: {md5_checksum} but got {actual_md5_checksum}. Downloaded file could have been modified / corrupted, clean cache and retry...')

    def _open_url_and_load_gzip_file(url, data_home, n_retries, delay, arff_params):
        if False:
            i = 10
            return i + 15
        gzip_file = _open_openml_url(url, data_home, n_retries=n_retries, delay=delay)
        with closing(gzip_file):
            return load_arff_from_gzip_file(gzip_file, **arff_params)
    arff_params: Dict = dict(parser=parser, output_type=output_type, openml_columns_info=openml_columns_info, feature_names_to_select=feature_names_to_select, target_names_to_select=target_names_to_select, shape=shape, read_csv_kwargs=read_csv_kwargs or {})
    try:
        (X, y, frame, categories) = _open_url_and_load_gzip_file(url, data_home, n_retries, delay, arff_params)
    except Exception as exc:
        if parser != 'pandas':
            raise
        from pandas.errors import ParserError
        if not isinstance(exc, ParserError):
            raise
        arff_params['read_csv_kwargs'].update(quotechar="'")
        (X, y, frame, categories) = _open_url_and_load_gzip_file(url, data_home, n_retries, delay, arff_params)
    return (X, y, frame, categories)

def _download_data_to_bunch(url: str, sparse: bool, data_home: Optional[str], *, as_frame: bool, openml_columns_info: List[dict], data_columns: List[str], target_columns: List[str], shape: Optional[Tuple[int, int]], md5_checksum: str, n_retries: int=3, delay: float=1.0, parser: str, read_csv_kwargs: Optional[Dict]=None):
    if False:
        for i in range(10):
            print('nop')
    'Download ARFF data, load it to a specific container and create to Bunch.\n\n    This function has a mechanism to retry/cache/clean the data.\n\n    Parameters\n    ----------\n    url : str\n        The URL of the ARFF file on OpenML.\n\n    sparse : bool\n        Whether the dataset is expected to use the sparse ARFF format.\n\n    data_home : str\n        The location where to cache the data.\n\n    as_frame : bool\n        Whether or not to return the data into a pandas DataFrame.\n\n    openml_columns_info : list of dict\n        The information regarding the columns provided by OpenML for the\n        ARFF dataset. The information is stored as a list of dictionaries.\n\n    data_columns : list of str\n        The list of the features to be selected.\n\n    target_columns : list of str\n        The list of the target variables to be selected.\n\n    shape : tuple or None\n        With `parser="liac-arff"`, when using a generator to load the data,\n        one needs to provide the shape of the data beforehand.\n\n    md5_checksum : str\n        The MD5 checksum provided by OpenML to check the data integrity.\n\n    n_retries : int, default=3\n        Number of retries when HTTP errors are encountered. Error with status\n        code 412 won\'t be retried as they represent OpenML generic errors.\n\n    delay : float, default=1.0\n        Number of seconds between retries.\n\n    parser : {"liac-arff", "pandas"}\n        The parser used to parse the ARFF file.\n\n    read_csv_kwargs : dict, default=None\n        Keyword arguments to pass to `pandas.read_csv` when using the pandas parser.\n        It allows to overwrite the default options.\n\n        .. versionadded:: 1.3\n\n    Returns\n    -------\n    data : :class:`~sklearn.utils.Bunch`\n        Dictionary-like object, with the following attributes.\n\n        X : {ndarray, sparse matrix, dataframe}\n            The data matrix.\n        y : {ndarray, dataframe, series}\n            The target.\n        frame : dataframe or None\n            A dataframe containing both `X` and `y`. `None` if\n            `output_array_type != "pandas"`.\n        categories : list of str or None\n            The names of the features that are categorical. `None` if\n            `output_array_type == "pandas"`.\n    '
    features_dict = {feature['name']: feature for feature in openml_columns_info}
    if sparse:
        output_type = 'sparse'
    elif as_frame:
        output_type = 'pandas'
    else:
        output_type = 'numpy'
    _verify_target_data_type(features_dict, target_columns)
    for name in target_columns:
        column_info = features_dict[name]
        n_missing_values = int(column_info['number_of_missing_values'])
        if n_missing_values > 0:
            raise ValueError(f"Target column '{column_info['name']}' has {n_missing_values} missing values. Missing values are not supported for target columns.")
    no_retry_exception = None
    if parser == 'pandas':
        from pandas.errors import ParserError
        no_retry_exception = ParserError
    (X, y, frame, categories) = _retry_with_clean_cache(url, data_home, no_retry_exception)(_load_arff_response)(url, data_home, parser=parser, output_type=output_type, openml_columns_info=features_dict, feature_names_to_select=data_columns, target_names_to_select=target_columns, shape=shape, md5_checksum=md5_checksum, n_retries=n_retries, delay=delay, read_csv_kwargs=read_csv_kwargs)
    return Bunch(data=X, target=y, frame=frame, categories=categories, feature_names=data_columns, target_names=target_columns)

def _verify_target_data_type(features_dict, target_columns):
    if False:
        print('Hello World!')
    if not isinstance(target_columns, list):
        raise ValueError('target_column should be list, got: %s' % type(target_columns))
    found_types = set()
    for target_column in target_columns:
        if target_column not in features_dict:
            raise KeyError(f"Could not find target_column='{target_column}'")
        if features_dict[target_column]['data_type'] == 'numeric':
            found_types.add(np.float64)
        else:
            found_types.add(object)
        if features_dict[target_column]['is_ignore'] == 'true':
            warn(f"target_column='{target_column}' has flag is_ignore.")
        if features_dict[target_column]['is_row_identifier'] == 'true':
            warn(f"target_column='{target_column}' has flag is_row_identifier.")
    if len(found_types) > 1:
        raise ValueError('Can only handle homogeneous multi-target datasets, i.e., all targets are either numeric or categorical.')

def _valid_data_column_names(features_list, target_columns):
    if False:
        print('Hello World!')
    valid_data_column_names = []
    for feature in features_list:
        if feature['name'] not in target_columns and feature['is_ignore'] != 'true' and (feature['is_row_identifier'] != 'true'):
            valid_data_column_names.append(feature['name'])
    return valid_data_column_names

@validate_params({'name': [str, None], 'version': [Interval(Integral, 1, None, closed='left'), StrOptions({'active'})], 'data_id': [Interval(Integral, 1, None, closed='left'), None], 'data_home': [str, os.PathLike, None], 'target_column': [str, list, None], 'cache': [bool], 'return_X_y': [bool], 'as_frame': [bool, StrOptions({'auto'})], 'n_retries': [Interval(Integral, 1, None, closed='left')], 'delay': [Interval(Real, 0, None, closed='right')], 'parser': [StrOptions({'auto', 'pandas', 'liac-arff'}), Hidden(StrOptions({'warn'}))], 'read_csv_kwargs': [dict, None]}, prefer_skip_nested_validation=True)
def fetch_openml(name: Optional[str]=None, *, version: Union[str, int]='active', data_id: Optional[int]=None, data_home: Optional[Union[str, os.PathLike]]=None, target_column: Optional[Union[str, List]]='default-target', cache: bool=True, return_X_y: bool=False, as_frame: Union[str, bool]='auto', n_retries: int=3, delay: float=1.0, parser: str='warn', read_csv_kwargs: Optional[Dict]=None):
    if False:
        while True:
            i = 10
    'Fetch dataset from openml by name or dataset id.\n\n    Datasets are uniquely identified by either an integer ID or by a\n    combination of name and version (i.e. there might be multiple\n    versions of the \'iris\' dataset). Please give either name or data_id\n    (not both). In case a name is given, a version can also be\n    provided.\n\n    Read more in the :ref:`User Guide <openml>`.\n\n    .. versionadded:: 0.20\n\n    .. note:: EXPERIMENTAL\n\n        The API is experimental (particularly the return value structure),\n        and might have small backward-incompatible changes without notice\n        or warning in future releases.\n\n    Parameters\n    ----------\n    name : str, default=None\n        String identifier of the dataset. Note that OpenML can have multiple\n        datasets with the same name.\n\n    version : int or \'active\', default=\'active\'\n        Version of the dataset. Can only be provided if also ``name`` is given.\n        If \'active\' the oldest version that\'s still active is used. Since\n        there may be more than one active version of a dataset, and those\n        versions may fundamentally be different from one another, setting an\n        exact version is highly recommended.\n\n    data_id : int, default=None\n        OpenML ID of the dataset. The most specific way of retrieving a\n        dataset. If data_id is not given, name (and potential version) are\n        used to obtain a dataset.\n\n    data_home : str or path-like, default=None\n        Specify another download and cache folder for the data sets. By default\n        all scikit-learn data is stored in \'~/scikit_learn_data\' subfolders.\n\n    target_column : str, list or None, default=\'default-target\'\n        Specify the column name in the data to use as target. If\n        \'default-target\', the standard target column a stored on the server\n        is used. If ``None``, all columns are returned as data and the\n        target is ``None``. If list (of strings), all columns with these names\n        are returned as multi-target (Note: not all scikit-learn classifiers\n        can handle all types of multi-output combinations).\n\n    cache : bool, default=True\n        Whether to cache the downloaded datasets into `data_home`.\n\n    return_X_y : bool, default=False\n        If True, returns ``(data, target)`` instead of a Bunch object. See\n        below for more information about the `data` and `target` objects.\n\n    as_frame : bool or \'auto\', default=\'auto\'\n        If True, the data is a pandas DataFrame including columns with\n        appropriate dtypes (numeric, string or categorical). The target is\n        a pandas DataFrame or Series depending on the number of target_columns.\n        The Bunch will contain a ``frame`` attribute with the target and the\n        data. If ``return_X_y`` is True, then ``(data, target)`` will be pandas\n        DataFrames or Series as describe above.\n\n        If `as_frame` is \'auto\', the data and target will be converted to\n        DataFrame or Series as if `as_frame` is set to True, unless the dataset\n        is stored in sparse format.\n\n        If `as_frame` is False, the data and target will be NumPy arrays and\n        the `data` will only contain numerical values when `parser="liac-arff"`\n        where the categories are provided in the attribute `categories` of the\n        `Bunch` instance. When `parser="pandas"`, no ordinal encoding is made.\n\n        .. versionchanged:: 0.24\n           The default value of `as_frame` changed from `False` to `\'auto\'`\n           in 0.24.\n\n    n_retries : int, default=3\n        Number of retries when HTTP errors or network timeouts are encountered.\n        Error with status code 412 won\'t be retried as they represent OpenML\n        generic errors.\n\n    delay : float, default=1.0\n        Number of seconds between retries.\n\n    parser : {"auto", "pandas", "liac-arff"}, default="liac-arff"\n        Parser used to load the ARFF file. Two parsers are implemented:\n\n        - `"pandas"`: this is the most efficient parser. However, it requires\n          pandas to be installed and can only open dense datasets.\n        - `"liac-arff"`: this is a pure Python ARFF parser that is much less\n          memory- and CPU-efficient. It deals with sparse ARFF datasets.\n\n        If `"auto"` (future default), the parser is chosen automatically such that\n        `"liac-arff"` is selected for sparse ARFF datasets, otherwise\n        `"pandas"` is selected.\n\n        .. versionadded:: 1.2\n        .. versionchanged:: 1.4\n           The default value of `parser` will change from `"liac-arff"` to\n           `"auto"` in 1.4. You can set `parser="auto"` to silence this\n           warning. Therefore, an `ImportError` will be raised from 1.4 if\n           the dataset is dense and pandas is not installed.\n\n    read_csv_kwargs : dict, default=None\n        Keyword arguments passed to :func:`pandas.read_csv` when loading the data\n        from a ARFF file and using the pandas parser. It can allow to\n        overwrite some default parameters.\n\n        .. versionadded:: 1.3\n\n    Returns\n    -------\n    data : :class:`~sklearn.utils.Bunch`\n        Dictionary-like object, with the following attributes.\n\n        data : np.array, scipy.sparse.csr_matrix of floats, or pandas DataFrame\n            The feature matrix. Categorical features are encoded as ordinals.\n        target : np.array, pandas Series or DataFrame\n            The regression target or classification labels, if applicable.\n            Dtype is float if numeric, and object if categorical. If\n            ``as_frame`` is True, ``target`` is a pandas object.\n        DESCR : str\n            The full description of the dataset.\n        feature_names : list\n            The names of the dataset columns.\n        target_names: list\n            The names of the target columns.\n\n        .. versionadded:: 0.22\n\n        categories : dict or None\n            Maps each categorical feature name to a list of values, such\n            that the value encoded as i is ith in the list. If ``as_frame``\n            is True, this is None.\n        details : dict\n            More metadata from OpenML.\n        frame : pandas DataFrame\n            Only present when `as_frame=True`. DataFrame with ``data`` and\n            ``target``.\n\n    (data, target) : tuple if ``return_X_y`` is True\n\n        .. note:: EXPERIMENTAL\n\n            This interface is **experimental** and subsequent releases may\n            change attributes without notice (although there should only be\n            minor changes to ``data`` and ``target``).\n\n        Missing values in the \'data\' are represented as NaN\'s. Missing values\n        in \'target\' are represented as NaN\'s (numerical target) or None\n        (categorical target).\n\n    Notes\n    -----\n    The `"pandas"` and `"liac-arff"` parsers can lead to different data types\n    in the output. The notable differences are the following:\n\n    - The `"liac-arff"` parser always encodes categorical features as `str` objects.\n      To the contrary, the `"pandas"` parser instead infers the type while\n      reading and numerical categories will be casted into integers whenever\n      possible.\n    - The `"liac-arff"` parser uses float64 to encode numerical features\n      tagged as \'REAL\' and \'NUMERICAL\' in the metadata. The `"pandas"`\n      parser instead infers if these numerical features corresponds\n      to integers and uses panda\'s Integer extension dtype.\n    - In particular, classification datasets with integer categories are\n      typically loaded as such `(0, 1, ...)` with the `"pandas"` parser while\n      `"liac-arff"` will force the use of string encoded class labels such as\n      `"0"`, `"1"` and so on.\n    - The `"pandas"` parser will not strip single quotes - i.e. `\'` - from\n      string columns. For instance, a string `\'my string\'` will be kept as is\n      while the `"liac-arff"` parser will strip the single quotes. For\n      categorical columns, the single quotes are stripped from the values.\n\n    In addition, when `as_frame=False` is used, the `"liac-arff"` parser\n    returns ordinally encoded data where the categories are provided in the\n    attribute `categories` of the `Bunch` instance. Instead, `"pandas"` returns\n    a NumPy array were the categories are not encoded.\n    '
    if cache is False:
        data_home = None
    else:
        data_home = get_data_home(data_home=data_home)
        data_home = join(str(data_home), 'openml')
    if name is not None:
        name = name.lower()
        if data_id is not None:
            raise ValueError('Dataset data_id={} and name={} passed, but you can only specify a numeric data_id or a name, not both.'.format(data_id, name))
        data_info = _get_data_info_by_name(name, version, data_home, n_retries=n_retries, delay=delay)
        data_id = data_info['did']
    elif data_id is not None:
        if version != 'active':
            raise ValueError('Dataset data_id={} and version={} passed, but you can only specify a numeric data_id or a version, not both.'.format(data_id, version))
    else:
        raise ValueError('Neither name nor data_id are provided. Please provide name or data_id.')
    data_description = _get_data_description_by_id(data_id, data_home)
    if data_description['status'] != 'active':
        warn('Version {} of dataset {} is inactive, meaning that issues have been found in the dataset. Try using a newer version from this URL: {}'.format(data_description['version'], data_description['name'], data_description['url']))
    if 'error' in data_description:
        warn('OpenML registered a problem with the dataset. It might be unusable. Error: {}'.format(data_description['error']))
    if 'warning' in data_description:
        warn('OpenML raised a warning on the dataset. It might be unusable. Warning: {}'.format(data_description['warning']))
    if parser == 'warn':
        parser = 'liac-arff'
        warn("The default value of `parser` will change from `'liac-arff'` to `'auto'` in 1.4. You can set `parser='auto'` to silence this warning. Therefore, an `ImportError` will be raised from 1.4 if the dataset is dense and pandas is not installed. Note that the pandas parser may return different data types. See the Notes Section in fetch_openml's API doc for details.", FutureWarning)
    return_sparse = data_description['format'].lower() == 'sparse_arff'
    as_frame = not return_sparse if as_frame == 'auto' else as_frame
    if parser == 'auto':
        parser_ = 'liac-arff' if return_sparse else 'pandas'
    else:
        parser_ = parser
    if as_frame or parser_ == 'pandas':
        try:
            check_pandas_support('`fetch_openml`')
        except ImportError as exc:
            if as_frame:
                err_msg = "Returning pandas objects requires pandas to be installed. Alternatively, explicitly set `as_frame=False` and `parser='liac-arff'`."
                raise ImportError(err_msg) from exc
            else:
                err_msg = f"Using `parser={parser_!r}` requires pandas to be installed. Alternatively, explicitly set `parser='liac-arff'`."
                if parser == 'auto':
                    warn("From version 1.4, `parser='auto'` with `as_frame=False` will use pandas. Either install pandas or set explicitly `parser='liac-arff'` to preserve the current behavior.", FutureWarning)
                    parser_ = 'liac-arff'
                else:
                    raise ImportError(err_msg) from exc
    if return_sparse:
        if as_frame:
            raise ValueError("Sparse ARFF datasets cannot be loaded with as_frame=True. Use as_frame=False or as_frame='auto' instead.")
        if parser_ == 'pandas':
            raise ValueError(f"Sparse ARFF datasets cannot be loaded with parser={parser!r}. Use parser='liac-arff' or parser='auto' instead.")
    features_list = _get_data_features(data_id, data_home)
    if not as_frame:
        for feature in features_list:
            if 'true' in (feature['is_ignore'], feature['is_row_identifier']):
                continue
            if feature['data_type'] == 'string':
                raise ValueError('STRING attributes are not supported for array representation. Try as_frame=True')
    if target_column == 'default-target':
        target_columns = [feature['name'] for feature in features_list if feature['is_target'] == 'true']
    elif isinstance(target_column, str):
        target_columns = [target_column]
    elif target_column is None:
        target_columns = []
    else:
        target_columns = target_column
    data_columns = _valid_data_column_names(features_list, target_columns)
    shape: Optional[Tuple[int, int]]
    if not return_sparse:
        data_qualities = _get_data_qualities(data_id, data_home)
        shape = (_get_num_samples(data_qualities), len(features_list))
    else:
        shape = None
    url = _DATA_FILE.format(data_description['file_id'])
    bunch = _download_data_to_bunch(url, return_sparse, data_home, as_frame=bool(as_frame), openml_columns_info=features_list, shape=shape, target_columns=target_columns, data_columns=data_columns, md5_checksum=data_description['md5_checksum'], n_retries=n_retries, delay=delay, parser=parser_, read_csv_kwargs=read_csv_kwargs)
    if return_X_y:
        return (bunch.data, bunch.target)
    description = '{}\n\nDownloaded from openml.org.'.format(data_description.pop('description'))
    bunch.update(DESCR=description, details=data_description, url='https://www.openml.org/d/{}'.format(data_id))
    return bunch