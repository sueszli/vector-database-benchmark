from localstack import config
from localstack.http import Request
from localstack.services.internal import DiagnoseResource

def test_diagnose_resource():
    if False:
        for i in range(10):
            print('nop')
    resource = DiagnoseResource()
    result = resource.on_get(Request(path='/_localstack/diagnose'))
    assert '/tmp' in result['file-tree']
    assert '/var/lib/localstack' in result['file-tree']
    assert result['config']['DATA_DIR'] == config.DATA_DIR
    assert result['config']['GATEWAY_LISTEN'] == [config.HostAndPort('0.0.0.0', 4566)]
    assert result['important-endpoints']['localhost.localstack.cloud'].startswith('127.0.')
    assert result['logs']['docker']