"""
Implementation of the data to json mapper
"""
import json
from typing import Dict
from samcli.lib.list.list_interfaces import Mapper

class DataToJsonMapper(Mapper):

    def map(self, data: Dict[str, str]) -> str:
        if False:
            i = 10
            return i + 15
        output = json.dumps(data, indent=2)
        return output