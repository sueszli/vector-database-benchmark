"""
Implementation of the resources to table mapper
"""
from collections import OrderedDict
from typing import Any, Dict
from samcli.lib.list.list_interfaces import Mapper

class ResourcesToTableMapper(Mapper):
    """
    Mapper class for mapping resources data for table output
    """

    def map(self, data: list) -> Dict[Any, Any]:
        if False:
            return 10
        '\n        Maps data to the format needed for consumption by the table consumer\n\n        Parameters\n        ----------\n        data: list\n            List of dictionaries containing the entries of the resources data\n\n        Returns\n        -------\n        table_data: Dict[Any, Any]\n            Dictionary containing the information and data needed for the table\n            consumer to output the data in table format\n        '
        entry_list = []
        for resource in data:
            entry_list.append([resource.get('LogicalResourceId', '-'), resource.get('PhysicalResourceId', '-')])
        table_data = {'format_string': '{Logical ID:<{0}} {Physical ID:<{1}}', 'format_args': OrderedDict({'Logical ID': 'Logical ID', 'Physical ID': 'Physical ID'}), 'table_name': 'Resources', 'data': entry_list}
        return table_data