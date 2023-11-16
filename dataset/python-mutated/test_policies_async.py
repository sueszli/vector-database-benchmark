from azure.mgmt.core import AsyncARMPipelineClient
from azure.mgmt.core.policies import ARMHttpLoggingPolicy
from azure.core.configuration import Configuration

def test_default_http_logging_policy():
    if False:
        print('Hello World!')
    config = Configuration()
    pipeline_client = AsyncARMPipelineClient(base_url='test', config=config)
    http_logging_policy = pipeline_client._pipeline._impl_policies[-1]._policy
    assert http_logging_policy.allowed_header_names == ARMHttpLoggingPolicy.DEFAULT_HEADERS_WHITELIST

def test_pass_in_http_logging_policy():
    if False:
        return 10
    config = Configuration()
    http_logging_policy = ARMHttpLoggingPolicy()
    http_logging_policy.allowed_header_names.update({'x-ms-added-header'})
    config.http_logging_policy = http_logging_policy
    pipeline_client = AsyncARMPipelineClient(base_url='test', config=config)
    http_logging_policy = pipeline_client._pipeline._impl_policies[-1]._policy
    assert http_logging_policy.allowed_header_names == ARMHttpLoggingPolicy.DEFAULT_HEADERS_WHITELIST.union({'x-ms-added-header'})