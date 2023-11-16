"""
The table consumer for 'sam list'
"""
from typing import Any, Dict
from samcli.commands._utils.table_print import pprint_column_names, pprint_columns
from samcli.lib.list.list_interfaces import ListInfoPullerConsumer

class StringConsumerTableOutput(ListInfoPullerConsumer):
    """
    Outputs data in table format
    """

    def consume(self, data: Dict[Any, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Outputs the data in a table format\n        Parameters\n        ----------\n        data: Dict[Any, Any]\n            The data to be outputted\n        '

        @pprint_column_names(format_string=data['format_string'], format_kwargs=data['format_args'], table_header=data['table_name'])
        def print_table_rows(**kwargs):
            if False:
                while True:
                    i = 10
            '\n            Prints the rows of the table based on the data provided\n            '
            for entry in data['data']:
                pprint_columns(columns=entry, width=kwargs['width'], margin=kwargs['margin'], format_string=data['format_string'], format_args=kwargs['format_args'], columns_dict=data['format_args'].copy())
        print_table_rows()