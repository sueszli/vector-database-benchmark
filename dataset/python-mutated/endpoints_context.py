"""
Display of the Endpoints of a SAM stack
"""
import logging
from typing import Optional
from samcli.commands.list.cli_common.list_common_context import ListContext
from samcli.lib.list.endpoints.endpoints_producer import EndpointsProducer
from samcli.lib.list.list_interfaces import ProducersEnum
from samcli.lib.list.mapper_consumer_factory import MapperConsumerFactory
LOG = logging.getLogger(__name__)

class EndpointsContext(ListContext):
    """
    Context class for sam list endpoints
    """

    def __init__(self, stack_name: str, output: str, region: Optional[str], profile: Optional[str], template_file: Optional[str], parameter_overrides: Optional[dict]=None):
        if False:
            return 10
        '\n        Parameters\n        ----------\n        stack_name: str\n            The name of the stack\n        output: str\n            The format of the output, either json or table\n        region: Optional[str]\n            The region of the stack\n        profile: Optional[str]\n            Optional profile to be used\n        template_file: Optional[str]\n            The location of the template file. If one is not specified, the default will be "template.yaml" in the CWD\n        parameter_overrides: Optional[dict]\n            Dictionary of parameters to override in the template\n        '
        super().__init__()
        self.parameter_overrides = parameter_overrides
        self.stack_name = stack_name
        self.output = output
        self.region = region
        self.profile = profile
        self.template_file = template_file
        self.iam_client = None
        self.cloudcontrol_client = None
        self.apigateway_client = None
        self.apigatewayv2_client = None

    def __enter__(self):
        if False:
            while True:
                i = 10
        self.init_clients()
        return self

    def __exit__(self, *args):
        if False:
            while True:
                i = 10
        pass

    def init_clients(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Initialize the clients being used by sam list.\n        '
        super().init_clients()
        self.iam_client = self.client_provider('iam')
        self.cloudcontrol_client = self.client_provider('cloudcontrol')
        self.apigateway_client = self.client_provider('apigateway')
        self.apigatewayv2_client = self.client_provider('apigatewayv2')

    def run(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the resources for a stack\n        '
        factory = MapperConsumerFactory()
        container = factory.create(producer=ProducersEnum.ENDPOINTS_PRODUCER, output=self.output)
        endpoints_producer = EndpointsProducer(stack_name=self.stack_name, region=self.region, profile=self.profile, template_file=self.template_file, cloudformation_client=self.cloudformation_client, iam_client=self.iam_client, cloudcontrol_client=self.cloudcontrol_client, apigateway_client=self.apigateway_client, apigatewayv2_client=self.apigatewayv2_client, mapper=container.mapper, consumer=container.consumer, parameter_overrides=self.parameter_overrides)
        endpoints_producer.produce()