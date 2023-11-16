"""
Implementation of the stack output to table mapper
"""
from collections import OrderedDict
from typing import Any, Dict
from samcli.lib.list.list_interfaces import Mapper

class StackOutputToTableMapper(Mapper):
    """
    Mapper class for mapping stack-outputs data for table output
    """

    def map(self, data: list) -> Dict[Any, Any]:
        if False:
            while True:
                i = 10
        '\n        Maps data to the format needed for consumption by the table consumer\n\n        Parameters\n        ----------\n        data: list\n            List of dictionaries containing the entries of the stack outputs data\n\n        Returns\n        -------\n        table_data: Dict[Any, Any]\n            Dictionary containing the information and data needed for the table consumer\n            to output the data in table format\n        '
        entry_list = []
        for stack_output in data:
            entry_list.append([stack_output.get('OutputKey', '-'), stack_output.get('OutputValue', '-'), stack_output.get('Description', '-')])
        table_data = {'format_string': '{OutputKey:<{0}} {OutputValue:<{1}} {Description:<{2}}', 'format_args': OrderedDict({'OutputKey': 'OutputKey', 'OutputValue': 'OutputValue', 'Description': 'Description'}), 'table_name': 'Stack Outputs', 'data': entry_list}
        return table_data