"""
Implementation of the endpoints to table mapper
"""
from collections import OrderedDict
from typing import Any, Dict
from samcli.lib.list.list_interfaces import Mapper
NO_DATA = '-'
SPACING = ''
CLOUD_ENDPOINT = 'CloudEndpoint'
METHODS = 'Methods'

class EndpointsToTableMapper(Mapper):
    """
    Mapper class for mapping endpoints data for table output
    """

    def map(self, data: list) -> Dict[Any, Any]:
        if False:
            i = 10
            return i + 15
        '\n        Maps data to the format needed for consumption by the table consumer\n\n        Parameters\n        ----------\n        data: list\n            List of dictionaries containing the entries of the endpoints data\n\n        Returns\n        -------\n        table_data: Dict[Any, Any]\n            Dictionary containing the information and data needed for the table consumer\n            to output the data in table format\n        '
        entry_list = []
        for endpoint in data:
            cloud_endpoint_furl_string = endpoint.get(CLOUD_ENDPOINT, NO_DATA)
            methods_string = NO_DATA
            cloud_endpoint_furl_multi_list = []
            if isinstance(endpoint.get(CLOUD_ENDPOINT, NO_DATA), list) and endpoint.get(CLOUD_ENDPOINT, []):
                cloud_endpoint_furl_string = endpoint.get(CLOUD_ENDPOINT, [NO_DATA])[0]
                if len(endpoint.get(CLOUD_ENDPOINT, [])) > 1:
                    cloud_endpoint_furl_multi_list = endpoint.get(CLOUD_ENDPOINT, [SPACING, SPACING])[1:]
            if isinstance(endpoint.get(METHODS, NO_DATA), list) and endpoint.get(METHODS, []):
                methods_string = '; '.join(endpoint.get(METHODS, []))
            entry_list.append([endpoint.get('LogicalResourceId', NO_DATA), endpoint.get('PhysicalResourceId', NO_DATA), cloud_endpoint_furl_string, methods_string])
            if cloud_endpoint_furl_multi_list:
                for url in cloud_endpoint_furl_multi_list:
                    entry_list.append([SPACING, SPACING, url, SPACING])
        table_data = {'format_string': '{Resource ID:<{0}} {Physical ID:<{1}} {Cloud Endpoints:<{2}} {Methods:<{3}}', 'format_args': OrderedDict({'Resource ID': 'Resource ID', 'Physical ID': 'Physical ID', 'Cloud Endpoints': 'Cloud Endpoints', 'Methods': 'Methods'}), 'table_name': 'Endpoints', 'data': entry_list}
        return table_data