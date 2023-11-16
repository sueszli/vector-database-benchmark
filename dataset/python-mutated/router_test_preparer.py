import os
from devtools_testutils import is_live, is_live_and_not_recording
from azure.communication.jobrouter._shared.utils import parse_connection_str

def router_test_decorator(func, **kwargs):
    if False:
        i = 10
        return i + 15

    def wrapper(self, *args, **kwargs):
        if False:
            return 10
        if is_live() or is_live_and_not_recording():
            self.connection_string = os.getenv('COMMUNICATION_LIVETEST_DYNAMIC_CONNECTION_STRING')
            (endpoint, _) = parse_connection_str(self.connection_string)
            self.resource_name = endpoint.split('.')[0]
        else:
            self.connection_string = 'endpoint=https://sanitized.communication.azure.net/;accesskey=fake==='
            self.resource_name = 'sanitized'
        func(self, *args, **kwargs)
    return wrapper