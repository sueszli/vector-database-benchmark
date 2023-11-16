from types import ModuleType
import pytest
from click.testing import CliRunner
from localstack.aws.scaffold import generate
from localstack.testing.pytest import markers

@markers.skip_offline
@pytest.mark.parametrize('service', ['apigateway', 'autoscaling', 'cloudformation', 'dynamodb', 'glue', 'kafka', 'kinesis', 'sqs', 's3'])
def test_generated_code_compiles(service, caplog):
    if False:
        i = 10
        return i + 15
    caplog.set_level(100000)
    runner = CliRunner()
    result = runner.invoke(generate, [service, '--no-doc', '--print'])
    assert result.exit_code == 0
    code = result.output
    compiled = compile(code, '<string>', 'exec')
    module = ModuleType(service)
    exec(compiled, module.__dict__)