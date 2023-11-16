"""Lambda scenario tests for different runtimes (i.e., multiruntime tests).

Directly correlates to the structure found in tests.aws.lambda_.functions.common
Each scenario has the following folder structure: ./common/<scenario>/runtime/
Runtime can either be directly one of the supported runtimes (e.g. in case of version specific compilation instructions)
or one of the keys in RUNTIMES_AGGREGATED. To selectively execute runtimes, use the runtimes parameter of multiruntime.
Example: runtimes=[Runtime.go1_x]
"""
import json
import logging
import time
import zipfile
import pytest
from localstack.testing.pytest import markers
from localstack.testing.snapshots.transformer import KeyValueBasedTransformer
from localstack.utils.files import cp_r
from localstack.utils.platform import get_arch
from localstack.utils.strings import short_uid, to_bytes, to_str
LOG = logging.getLogger(__name__)

@pytest.fixture(autouse=True)
def snapshot_transformers(snapshot):
    if False:
        while True:
            i = 10
    snapshot.add_transformer(snapshot.transform.lambda_api())
    snapshot.add_transformers_list([snapshot.transform.key_value('AWS_ACCESS_KEY_ID', 'aws-access-key-id'), snapshot.transform.key_value('AWS_SECRET_ACCESS_KEY', 'aws-secret-access-key'), snapshot.transform.key_value('AWS_SESSION_TOKEN', 'aws-session-token'), snapshot.transform.key_value('_X_AMZN_TRACE_ID', 'x-amzn-trace-id'), snapshot.transform.key_value('_LAMBDA_SERVER_PORT', '<lambda-server-port>', reference_replacement=False), KeyValueBasedTransformer(lambda k, v: str(v) if k == 'remaining_time_in_millis' else None, '<remaining-time-in-millis>', replace_reference=False), snapshot.transform.key_value('deadline', 'deadline')])

@pytest.mark.skipif(condition=get_arch() != 'x86_64', reason="build process doesn't support arm64 right now")
class TestLambdaRuntimesCommon:

    @markers.aws.validated
    @markers.multiruntime(scenario='echo')
    def test_echo_invoke(self, multiruntime_lambda, aws_client):
        if False:
            i = 10
            return i + 15
        create_function_result = multiruntime_lambda.create_function(MemorySize=1024, Timeout=5)

        def _invoke_with_payload(payload):
            if False:
                while True:
                    i = 10
            invoke_result = aws_client.lambda_.invoke(FunctionName=create_function_result['FunctionName'], Payload=to_bytes(json.dumps(payload)))
            assert invoke_result['StatusCode'] == 200
            assert json.loads(invoke_result['Payload'].read()) == payload
            assert not invoke_result.get('FunctionError')
        payload = {'hello': 'world'}
        _invoke_with_payload(payload)
        payload = {'hello': '\'" some other \'\'"" quotes, a emoji 🥳 and some brackets {[}}[([]))'}
        _invoke_with_payload(payload)
        payload = {'hello': 'obi wan!' * 128 * 1024 * 5}
        _invoke_with_payload(payload)
        payload = True
        _invoke_with_payload(payload)
        payload = False
        _invoke_with_payload(payload)
        payload = None
        _invoke_with_payload(payload)
        payload = [1, 2]
        _invoke_with_payload(payload)
        payload = 1
        _invoke_with_payload(payload)
        invoke_result = aws_client.lambda_.invoke(FunctionName=create_function_result['FunctionName'])
        assert invoke_result['StatusCode'] == 200
        assert json.loads(invoke_result['Payload'].read()) == {}
        assert not invoke_result.get('FunctionError')

    @markers.snapshot.skip_snapshot_verify(paths=['$..environment.LOCALSTACK_HOSTNAME', '$..environment.EDGE_PORT', '$..environment.AWS_ENDPOINT_URL', '$..environment.AWS_LAMBDA_FUNCTION_TIMEOUT', '$..environment.AWS_CONTAINER_AUTHORIZATION_TOKEN', '$..environment.AWS_CONTAINER_CREDENTIALS_FULL_URI', '$..environment.AWS_XRAY_CONTEXT_MISSING', '$..environment.AWS_XRAY_DAEMON_ADDRESS', '$..environment._AWS_XRAY_DAEMON_ADDRESS', '$..environment._AWS_XRAY_DAEMON_PORT', '$..environment._X_AMZN_TRACE_ID', '$..environment.NODE_EXTRA_CA_CERTS', '$..environment._LAMBDA_TELEMETRY_LOG_FD', '$..environment.AWS_EXECUTION_ENV', '$..environment.LD_LIBRARY_PATH', '$..environment.PATH', '$..CodeSha256'])
    @markers.aws.validated
    @markers.multiruntime(scenario='introspection')
    def test_introspection_invoke(self, multiruntime_lambda, snapshot, aws_client):
        if False:
            print('Hello World!')
        create_function_result = multiruntime_lambda.create_function(MemorySize=1024, Environment={'Variables': {'TEST_KEY': 'TEST_VAL'}})
        snapshot.match('create_function_result', create_function_result)
        invoke_result = aws_client.lambda_.invoke(FunctionName=create_function_result['FunctionName'], Payload=b'{"simple": "payload"}')
        assert invoke_result['StatusCode'] == 200
        invocation_result_payload = to_str(invoke_result['Payload'].read())
        invocation_result_payload = json.loads(invocation_result_payload)
        assert 'environment' in invocation_result_payload
        assert 'ctx' in invocation_result_payload
        assert 'packages' in invocation_result_payload
        snapshot.match('invocation_result_payload', invocation_result_payload)
        invoke_result_qualified = aws_client.lambda_.invoke(FunctionName=f"{create_function_result['FunctionArn']}:$LATEST", Payload=b'{"simple": "payload"}')
        assert invoke_result['StatusCode'] == 200
        invocation_result_payload_qualified = to_str(invoke_result_qualified['Payload'].read())
        invocation_result_payload_qualified = json.loads(invocation_result_payload_qualified)
        snapshot.match('invocation_result_payload_qualified', invocation_result_payload_qualified)

    @markers.snapshot.skip_snapshot_verify(paths=['$..CodeSha256'])
    @markers.aws.validated
    @markers.multiruntime(scenario='uncaughtexception')
    def test_uncaught_exception_invoke(self, multiruntime_lambda, snapshot, aws_client):
        if False:
            for i in range(10):
                print('nop')
        snapshot.add_transformer(snapshot.transform.key_value('stackTrace', '<stack-trace>', reference_replacement=False))
        snapshot.add_transformer(snapshot.transform.key_value('trace', '<stack-trace>', reference_replacement=False))
        create_function_result = multiruntime_lambda.create_function(MemorySize=1024)
        snapshot.match('create_function_result', create_function_result)
        invocation_result = aws_client.lambda_.invoke(FunctionName=create_function_result['FunctionName'], Payload=b'{"error_msg": "some_error_msg"}')
        assert 'FunctionError' in invocation_result
        snapshot.match('error_result', invocation_result)

    @markers.snapshot.skip_snapshot_verify(paths=['$..CodeSha256'])
    @markers.aws.validated
    @markers.multiruntime(scenario='introspection', runtimes=['nodejs'])
    def test_runtime_wrapper_invoke(self, multiruntime_lambda, snapshot, tmp_path, aws_client):
        if False:
            print('Hello World!')
        modified_zip = str(tmp_path / f'temp-zip-{short_uid()}.zip')
        cp_r(multiruntime_lambda.zip_file_path, modified_zip)
        test_value = f'test-value-{short_uid()}'
        env_wrapper = f'#!/bin/bash\n          export WRAPPER_VAR={test_value}\n          exec "$@"\n        '
        with zipfile.ZipFile(modified_zip, mode='a') as zip_file:
            info = zipfile.ZipInfo('environment_wrapper')
            info.date_time = time.localtime()
            info.external_attr = 33261 << 16
            zip_file.writestr(info, env_wrapper)
        multiruntime_lambda.zip_file_path = modified_zip
        create_function_result = multiruntime_lambda.create_function(MemorySize=1024, Environment={'Variables': {'AWS_LAMBDA_EXEC_WRAPPER': '/var/task/environment_wrapper'}})
        snapshot.match('create_function_result', create_function_result)
        invoke_result = aws_client.lambda_.invoke(FunctionName=create_function_result['FunctionName'], Payload=b'{"simple": "payload"}')
        assert invoke_result['StatusCode'] == 200
        invocation_result_payload = to_str(invoke_result['Payload'].read())
        invocation_result_payload = json.loads(invocation_result_payload)
        assert 'environment' in invocation_result_payload
        assert 'ctx' in invocation_result_payload
        assert 'packages' in invocation_result_payload
        assert invocation_result_payload['environment']['WRAPPER_VAR'] == test_value

@pytest.mark.skipif(condition=get_arch() != 'x86_64', reason="build process doesn't support arm64 right now")
class TestLambdaCallingLocalstack:

    @markers.multiruntime(scenario='endpointinjection', runtimes=['nodejs', 'python', 'ruby', 'java8.al2', 'java11', 'go1.x', 'dotnet6'])
    @markers.aws.only_localstack
    def test_calling_localstack_from_lambda(self, multiruntime_lambda, tmp_path, aws_client):
        if False:
            for i in range(10):
                print('nop')
        create_function_result = multiruntime_lambda.create_function(MemorySize=1024, Environment={'Variables': {'CONFIGURE_CLIENT': '1'}})
        invocation_result = aws_client.lambda_.invoke(FunctionName=create_function_result['FunctionName'], Payload=b'{}')
        assert 'FunctionError' not in invocation_result