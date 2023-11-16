from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, List, Mapping, Union
import requests

@dataclass
class Decoder:
    """
    Decoder strategy to transform a requests.Response into a Mapping[str, Any]
    """

    @abstractmethod
    def decode(self, response: requests.Response) -> Union[Mapping[str, Any], List]:
        if False:
            while True:
                i = 10
        '\n        Decodes a requests.Response into a Mapping[str, Any] or an array\n        :param response: the response to decode\n        :return: Mapping or array describing the response\n        '
        pass