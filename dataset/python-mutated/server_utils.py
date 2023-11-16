import json
import os
import tempfile
from typing import Any, Dict, Union
import numpy as np
import pandas as pd
from starlette.datastructures import UploadFile
from starlette.responses import JSONResponse
from ludwig.utils.data_utils import NumpyEncoder

def serialize_payload(data_source: Union[pd.DataFrame, pd.Series]) -> tuple:
    if False:
        for i in range(10):
            print('nop')
    "\n    Generates two dictionaries to be sent via REST API for Ludwig prediction\n    service.\n    First dictionary created is payload_dict. Keys found in payload_dict:\n    raw_data: this is json string created by pandas to_json() method\n    source_type: indicates if the data_source is either a pandas dataframe or\n        pandas series.  This is needed to know how to rebuild the structure.\n    ndarray_dtype:  this is a dictionary where each entry is for any ndarray\n        data found in the data_source.  This could be an empty dictioinary if no\n        ndarray objects are present in data_source. Key for this dictionary is\n        column name if data_source is dataframe or index name if data_source is\n        series.  The value portion of the dictionary is the dtype of the\n        ndarray.  This value is used to set the correct dtype when rebuilding\n        the entry.\n\n    Second dictionary created is called payload_files, this contains information\n    and content for files to be sent to the server.  NOTE: if no files are to be\n    sent, this will be an empty dictionary.\n    Entries in this dictionary:\n    Key: file path string for file to be sent to server\n    Value: tuple(file path string, byte encoded file content,\n                 'application/octet-stream')\n\n    Args:\n        data_source: input features to be sent to Ludwig server\n\n    Returns: tuple(payload_dict, payload_files)\n\n    "
    payload_dict = {}
    payload_dict['ndarray_dtype'] = {}
    payload_files = {}
    if isinstance(data_source, pd.DataFrame):
        payload_dict['raw_data'] = data_source.to_json(orient='columns')
        payload_dict['source_type'] = 'dataframe'
        for col in data_source.columns:
            if isinstance(data_source[col].iloc[0], np.ndarray):
                payload_dict['ndarray_dtype'][col] = str(data_source[col].iloc[0].dtype)
            elif isinstance(data_source[col].iloc[0], str) and os.path.exists(data_source[col].iloc[0]):
                for v in data_source[col]:
                    payload_files[v] = (v, open(v, 'rb'), 'application/octet-stream')
    elif isinstance(data_source, pd.Series):
        payload_dict['raw_data'] = data_source.to_json(orient='index')
        payload_dict['source_type'] = 'series'
        for col in data_source.index:
            if isinstance(data_source[col], np.ndarray):
                payload_dict['ndarray_dtype'][col] = str(data_source[col].dtype)
            elif isinstance(data_source[col], str) and os.path.exists(data_source[col]):
                v = data_source[col]
                payload_files[v] = (v, open(v, 'rb'), 'application/octet-stream')
    else:
        ValueError('"data_source" must be either a pandas DataFrame or Series, format found to be {}'.format(type(data_source)))
    return (payload_dict, payload_files)

def _write_file(v, files):
    if False:
        for i in range(10):
            print('nop')
    suffix = os.path.splitext(v.filename)[1]
    named_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    files.append(named_file)
    named_file.write(v.file.read())
    named_file.close()
    return named_file.name

def deserialize_payload(json_string: str) -> pd.DataFrame:
    if False:
        return 10
    'This function performs the inverse of the serialize_payload function and rebuilds the object represented in\n    json_string to a pandas DataFrame.\n\n    Args:\n        json_string: representing object to be rebuilt.\n\n    Returns: pandas.DataFrame\n    '
    payload_dict = json.loads(json_string)
    raw_data_dict = json.loads(payload_dict['raw_data'])
    if payload_dict['source_type'] == 'dataframe':
        df = pd.DataFrame.from_dict(raw_data_dict, orient='columns')
    elif payload_dict['source_type'] == 'series':
        df = pd.DataFrame(pd.Series(raw_data_dict)).T
    else:
        ValueError('Unknown "source_type" found.  Valid values are "dataframe" or "series".  Instead found {}'.format(payload_dict['source_type']))
    if payload_dict['ndarray_dtype']:
        for col in payload_dict['ndarray_dtype']:
            dtype = payload_dict['ndarray_dtype'][col]
            df[col] = df[col].apply(lambda x: np.array(x).astype(dtype))
    return df

def deserialize_request(form) -> tuple:
    if False:
        while True:
            i = 10
    'This function will deserialize the REST API request packet to create a pandas dataframe that is input to the\n    Ludwig predict method and a list of files that will be cleaned up at the end of processing.\n\n    Args:\n        form: REST API provide form data\n\n    Returns: tuple(pandas.DataFrame, list of temporary files to clean up)\n    '
    files = []
    file_index = {}
    for (k, v) in form.multi_items():
        if type(v) == UploadFile:
            file_index[v.filename] = _write_file(v, files)
    df = deserialize_payload(form['payload'])
    df.replace(to_replace=list(file_index.keys()), value=list(file_index.values()), inplace=True)
    return (df, files)

class NumpyJSONResponse(JSONResponse):

    def render(self, content: Dict[str, Any]) -> str:
        if False:
            while True:
                i = 10
        'Override the default JSONResponse behavior to encode numpy arrays.\n\n        Args:\n            content: JSON object to be serialized.\n\n        Returns: str\n        '
        return json.dumps(content, ensure_ascii=False, allow_nan=False, indent=None, separators=(',', ':'), cls=NumpyEncoder).encode('utf-8')