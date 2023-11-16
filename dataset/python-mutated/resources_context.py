"""
Display the Resources of a SAM stack
"""
import logging
from typing import Optional
from samcli.commands.list.cli_common.list_common_context import ListContext
from samcli.lib.list.list_interfaces import ProducersEnum
from samcli.lib.list.mapper_consumer_factory import MapperConsumerFactory
from samcli.lib.list.resources.resource_mapping_producer import ResourceMappingProducer
LOG = logging.getLogger(__name__)

class ResourcesContext(ListContext):

    def __init__(self, stack_name: str, output: str, region: Optional[str], profile: Optional[str], template_file: Optional[str], parameter_overrides: Optional[dict]=None):
        if False:
            return 10
        super().__init__()
        self.stack_name = stack_name
        self.output = output
        self.region = region
        self.profile = profile
        self.template_file = template_file
        self.iam_client = None
        self.parameter_overrides = parameter_overrides

    def __enter__(self):
        if False:
            while True:
                i = 10
        self.init_clients()
        return self

    def __exit__(self, *args):
        if False:
            return 10
        pass

    def init_clients(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Initialize the clients being used by sam list.\n        '
        super().init_clients()
        self.iam_client = self.client_provider('iam')

    def run(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the resources for a stack\n        '
        factory = MapperConsumerFactory()
        container = factory.create(producer=ProducersEnum.RESOURCES_PRODUCER, output=self.output)
        resource_producer = ResourceMappingProducer(stack_name=self.stack_name, region=self.region, profile=self.profile, template_file=self.template_file, cloudformation_client=self.cloudformation_client, iam_client=self.iam_client, mapper=container.mapper, consumer=container.consumer, parameter_overrides=self.parameter_overrides)
        resource_producer.produce()