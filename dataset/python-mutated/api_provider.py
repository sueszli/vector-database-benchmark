"""Class that provides the Api with a list of routes from a Template"""
import logging
from typing import Iterator, List, Optional
from samcli.lib.providers.api_collector import ApiCollector
from samcli.lib.providers.cfn_api_provider import CfnApiProvider
from samcli.lib.providers.cfn_base_api_provider import CfnBaseApiProvider
from samcli.lib.providers.provider import AbstractApiProvider, Api, Stack
from samcli.lib.providers.sam_api_provider import SamApiProvider
LOG = logging.getLogger(__name__)

class ApiProvider(AbstractApiProvider):

    def __init__(self, stacks: List[Stack], cwd: Optional[str]=None):
        if False:
            return 10
        '\n        Initialize the class with template data. The template_dict is assumed\n        to be valid, normalized and a dictionary. template_dict should be normalized by running any and all\n        pre-processing before passing to this class.\n        This class does not perform any syntactic validation of the template.\n\n        After the class is initialized, changes to ``template_dict`` will not be reflected in here.\n        You will need to explicitly update the class with new template, if necessary.\n\n        Parameters\n        ----------\n        stacks : dict\n            List of stacks apis are extracted from\n        cwd : str\n            Optional working directory with respect to which we will resolve relative path to Swagger file\n        '
        self.stacks = stacks
        self.cwd = cwd
        self.api = self._extract_api()
        self.routes = self.api.routes
        LOG.debug('%d APIs found in the template', len(self.routes))

    def get_all(self) -> Iterator[Api]:
        if False:
            while True:
                i = 10
        '\n        Yields all the Apis in the current Provider\n\n        :yields api: an Api object with routes and properties\n        '
        yield self.api

    def _extract_api(self) -> Api:
        if False:
            print('Hello World!')
        '\n        Extracts all the routes by running through the one providers. The provider that has the first type matched\n        will be run across all the resources\n\n        Parameters\n        ----------\n        Returns\n        ---------\n        An Api from the parsed template\n        '
        collector = ApiCollector()
        provider = ApiProvider.find_api_provider(self.stacks)
        provider.extract_resources(self.stacks, collector, cwd=self.cwd)
        return collector.get_api()

    @staticmethod
    def find_api_provider(stacks: List[Stack]) -> CfnBaseApiProvider:
        if False:
            i = 10
            return i + 15
        '\n        Finds the ApiProvider given the first api type of the resource\n\n        Parameters\n        -----------\n        stacks: List[Stack]\n            List of stacks apis are extracted from\n\n        Return\n        ----------\n        Instance of the ApiProvider that will be run on the template with a default of SamApiProvider\n        '
        for stack in stacks:
            for (_, resource) in stack.resources.items():
                if resource.get(CfnBaseApiProvider.RESOURCE_TYPE) in SamApiProvider.TYPES:
                    return SamApiProvider()
                if resource.get(CfnBaseApiProvider.RESOURCE_TYPE) in CfnApiProvider.TYPES:
                    return CfnApiProvider()
        return SamApiProvider()