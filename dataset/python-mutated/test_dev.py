import inspect
import watchfiles
from typer import Option
import prefect
from prefect.cli.dev import agent_process_entrypoint, start_agent
from prefect.testing.cli import invoke_and_assert
from prefect.testing.utilities import AsyncMock, MagicMock

def test_dev_start_runs_all_services(monkeypatch):
    if False:
        return 10
    "\n    Test that `prefect dev start` runs all services. This test mocks out the\n    `run_process` function along with the `watchfiles.arun_process` function\n    so the test doesn't actually start any processes; instead, it verifies that\n    the command attempts to start all services correctly.\n    "
    mock_run_process = AsyncMock()

    def mock_run_process_call(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if 'task_status' in kwargs:
            kwargs['task_status'].started()
    mock_run_process.side_effect = mock_run_process_call
    monkeypatch.setattr(prefect.cli.dev, 'run_process', mock_run_process)
    mock_arun_process = AsyncMock()
    monkeypatch.setattr(watchfiles, 'arun_process', mock_arun_process)
    mock_kill = MagicMock()
    monkeypatch.setattr(prefect.cli.dev.os, 'kill', mock_kill)
    mock_awatch = MagicMock()

    async def async_generator():
        yield None
    mock_awatch.return_value = async_generator()
    monkeypatch.setattr(watchfiles, 'awatch', mock_awatch)
    invoke_and_assert(['dev', 'start'], expected_code=0)
    mock_arun_process.assert_called_once()
    mock_run_process.assert_any_call(command=['npm', 'run', 'serve'], stream_output=True)
    uvicorn_called = False
    for call in mock_run_process.call_args_list:
        if 'command' in call.kwargs and 'uvicorn' in call.kwargs['command']:
            uvicorn_called = True
            break
    assert uvicorn_called

def test_agent_subprocess_entrypoint_runs_agent_with_valid_params(monkeypatch):
    if False:
        print('Hello World!')
    mock_agent_start = MagicMock()
    monkeypatch.setattr(prefect.cli.dev, 'start_agent', mock_agent_start)
    api = 'http://127.0.0.1:4200'
    work_queues = ['default']
    start_agent_signature = inspect.signature(start_agent)
    start_agent_params = start_agent_signature.parameters
    mock_agent_start.__signature__ = start_agent_signature
    agent_process_entrypoint(api=api, work_queues=work_queues)
    call_args = mock_agent_start.call_args[1]
    for (param_name, param) in start_agent_params.items():
        if hasattr(param.annotation, '__origin__'):
            arg_type = param.annotation.__origin__
        else:
            arg_type = param.annotation
        assert isinstance(call_args[param_name], arg_type) or call_args[param_name] is None
    assert call_args['api'] == api
    assert call_args['work_queues'] == work_queues

def test_mixed_parameter_default_types(monkeypatch):
    if False:
        while True:
            i = 10
    mock_agent_start = MagicMock()
    monkeypatch.setattr(prefect.cli.dev, 'start_agent', mock_agent_start)
    start_agent_signature = inspect.Signature(parameters=[inspect.Parameter(name='api', kind=inspect.Parameter.KEYWORD_ONLY, default=Option(None, '--api', help='The URL of the Prefect API server')), inspect.Parameter(name='work_queues', kind=inspect.Parameter.KEYWORD_ONLY, default=['default'])])
    mock_agent_start.__signature__ = start_agent_signature
    agent_process_entrypoint()
    mock_agent_start.assert_called_once()

def test_agent_subprocess_entrypoint_adds_typer_console(monkeypatch):
    if False:
        print('Hello World!')
    "\n    Ensures a Rich console is added to the PrefectTyper's global `app` instance.\n    "
    start_agent_signature = inspect.signature(start_agent)
    mock_agent_start = MagicMock()
    mock_agent_start.__signature__ = start_agent_signature
    monkeypatch.setattr(prefect.cli.dev, 'start_agent', mock_agent_start)
    mock_app = MagicMock()
    monkeypatch.setattr(prefect.cli.dev, 'app', mock_app)
    agent_process_entrypoint()
    assert mock_app.console is not None