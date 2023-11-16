"""
Common context class to inherit from for sam list sub-commands
"""
from samcli.lib.utils.boto_utils import get_boto_client_provider_with_config

class ListContext:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.cloudformation_client = None
        self.client_provider = None
        self.region = None
        self.profile = None

    def init_clients(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize the clients being used by sam list.\n        '
        from boto3 import Session
        if not self.region:
            session = Session()
            self.region = session.region_name
        client_provider = get_boto_client_provider_with_config(region=self.region, profile=self.profile)
        self.client_provider = client_provider
        self.cloudformation_client = client_provider('cloudformation')