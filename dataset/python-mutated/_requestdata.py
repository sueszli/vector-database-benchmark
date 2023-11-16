"""This module contains a class that holds the parameters of a request to the Bot API."""
import json
from typing import Any, Dict, List, Optional, Union, final
from urllib.parse import urlencode
from telegram._utils.types import UploadFileDict
from telegram.request._requestparameter import RequestParameter

@final
class RequestData:
    """Instances of this class collect the data needed for one request to the Bot API, including
    all parameters and files to be sent along with the request.

    .. versionadded:: 20.0

    Warning:
        How exactly instances of this are created should be considered an implementation detail
        and not part of PTBs public API. Users should exclusively rely on the documented
        attributes, properties and methods.

    Attributes:
        contains_files (:obj:`bool`): Whether this object contains files to be uploaded via
            ``multipart/form-data``.
    """
    __slots__ = ('_parameters', 'contains_files')

    def __init__(self, parameters: Optional[List[RequestParameter]]=None):
        if False:
            for i in range(10):
                print('nop')
        self._parameters: List[RequestParameter] = parameters or []
        self.contains_files: bool = any((param.input_files for param in self._parameters))

    @property
    def parameters(self) -> Dict[str, Union[str, int, List[Any], Dict[Any, Any]]]:
        if False:
            i = 10
            return i + 15
        'Gives the parameters as mapping of parameter name to the parameter value, which can be\n        a single object of type :obj:`int`, :obj:`float`, :obj:`str` or :obj:`bool` or any\n        (possibly nested) composition of lists, tuples and dictionaries, where each entry, key\n        and value is of one of the mentioned types.\n        '
        return {param.name: param.value for param in self._parameters if param.value is not None}

    @property
    def json_parameters(self) -> Dict[str, str]:
        if False:
            return 10
        "Gives the parameters as mapping of parameter name to the respective JSON encoded\n        value.\n\n        Tip:\n            By default, this property uses the standard library's :func:`json.dumps`.\n            To use a custom library for JSON encoding, you can directly encode the keys of\n            :attr:`parameters` - note that string valued keys should not be JSON encoded.\n        "
        return {param.name: param.json_value for param in self._parameters if param.json_value is not None}

    def url_encoded_parameters(self, encode_kwargs: Optional[Dict[str, Any]]=None) -> str:
        if False:
            while True:
                i = 10
        'Encodes the parameters with :func:`urllib.parse.urlencode`.\n\n        Args:\n            encode_kwargs (Dict[:obj:`str`, any], optional): Additional keyword arguments to pass\n                along to :func:`urllib.parse.urlencode`.\n        '
        if encode_kwargs:
            return urlencode(self.json_parameters, **encode_kwargs)
        return urlencode(self.json_parameters)

    def parametrized_url(self, url: str, encode_kwargs: Optional[Dict[str, Any]]=None) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Shortcut for attaching the return value of :meth:`url_encoded_parameters` to the\n        :paramref:`url`.\n\n        Args:\n            url (:obj:`str`): The URL the parameters will be attached to.\n            encode_kwargs (Dict[:obj:`str`, any], optional): Additional keyword arguments to pass\n                along to :func:`urllib.parse.urlencode`.\n        '
        url_parameters = self.url_encoded_parameters(encode_kwargs=encode_kwargs)
        return f'{url}?{url_parameters}'

    @property
    def json_payload(self) -> bytes:
        if False:
            while True:
                i = 10
        "The :attr:`parameters` as UTF-8 encoded JSON payload.\n\n        Tip:\n            By default, this property uses the standard library's :func:`json.dumps`.\n            To use a custom library for JSON encoding, you can directly encode the keys of\n            :attr:`parameters` - note that string valued keys should not be JSON encoded.\n        "
        return json.dumps(self.json_parameters).encode('utf-8')

    @property
    def multipart_data(self) -> UploadFileDict:
        if False:
            print('Hello World!')
        'Gives the files contained in this object as mapping of part name to encoded content.'
        multipart_data: UploadFileDict = {}
        for param in self._parameters:
            m_data = param.multipart_data
            if m_data:
                multipart_data.update(m_data)
        return multipart_data