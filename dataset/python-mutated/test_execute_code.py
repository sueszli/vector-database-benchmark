import random
import string
import tempfile
from pathlib import Path
import pytest
import autogpt.commands.execute_code as sut
from autogpt.agents.agent import Agent
from autogpt.agents.utils.exceptions import InvalidArgumentError, OperationNotAllowedError

@pytest.fixture
def random_code(random_string) -> str:
    if False:
        print('Hello World!')
    return f"print('Hello {random_string}!')"

@pytest.fixture
def python_test_file(agent: Agent, random_code: str):
    if False:
        i = 10
        return i + 15
    temp_file = tempfile.NamedTemporaryFile(dir=agent.workspace.root, suffix='.py')
    temp_file.write(str.encode(random_code))
    temp_file.flush()
    yield Path(temp_file.name)
    temp_file.close()

@pytest.fixture
def python_test_args_file(agent: Agent):
    if False:
        i = 10
        return i + 15
    temp_file = tempfile.NamedTemporaryFile(dir=agent.workspace.root, suffix='.py')
    temp_file.write(str.encode('import sys\nprint(sys.argv[1], sys.argv[2])'))
    temp_file.flush()
    yield Path(temp_file.name)
    temp_file.close()

@pytest.fixture
def random_string():
    if False:
        return 10
    return ''.join((random.choice(string.ascii_lowercase) for _ in range(10)))

def test_execute_python_file(python_test_file: Path, random_string: str, agent: Agent):
    if False:
        print('Hello World!')
    result: str = sut.execute_python_file(python_test_file, agent=agent)
    assert result.replace('\r', '') == f'Hello {random_string}!\n'

def test_execute_python_file_args(python_test_args_file: Path, random_string: str, agent: Agent):
    if False:
        for i in range(10):
            print('nop')
    random_args = [random_string] * 2
    random_args_string = ' '.join(random_args)
    result = sut.execute_python_file(python_test_args_file, args=random_args, agent=agent)
    assert result == f'{random_args_string}\n'

def test_execute_python_code(random_code: str, random_string: str, agent: Agent):
    if False:
        i = 10
        return i + 15
    result: str = sut.execute_python_code(random_code, agent=agent)
    assert result.replace('\r', '') == f'Hello {random_string}!\n'

def test_execute_python_file_invalid(agent: Agent):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(InvalidArgumentError):
        sut.execute_python_file('not_python', agent)

def test_execute_python_file_not_found(agent: Agent):
    if False:
        return 10
    with pytest.raises(FileNotFoundError, match="python: can't open file '([a-zA-Z]:)?[/\\\\\\-\\w]*notexist.py': \\[Errno 2\\] No such file or directory"):
        sut.execute_python_file('notexist.py', agent)

def test_execute_shell(random_string: str, agent: Agent):
    if False:
        while True:
            i = 10
    result = sut.execute_shell(f"echo 'Hello {random_string}!'", agent)
    assert f'Hello {random_string}!' in result

def test_execute_shell_local_commands_not_allowed(random_string: str, agent: Agent):
    if False:
        for i in range(10):
            print('nop')
    result = sut.execute_shell(f"echo 'Hello {random_string}!'", agent)
    assert f'Hello {random_string}!' in result

def test_execute_shell_denylist_should_deny(agent: Agent, random_string: str):
    if False:
        i = 10
        return i + 15
    agent.legacy_config.shell_denylist = ['echo']
    with pytest.raises(OperationNotAllowedError, match='not allowed'):
        sut.execute_shell(f"echo 'Hello {random_string}!'", agent)

def test_execute_shell_denylist_should_allow(agent: Agent, random_string: str):
    if False:
        print('Hello World!')
    agent.legacy_config.shell_denylist = ['cat']
    result = sut.execute_shell(f"echo 'Hello {random_string}!'", agent)
    assert 'Hello' in result and random_string in result

def test_execute_shell_allowlist_should_deny(agent: Agent, random_string: str):
    if False:
        i = 10
        return i + 15
    agent.legacy_config.shell_command_control = sut.ALLOWLIST_CONTROL
    agent.legacy_config.shell_allowlist = ['cat']
    with pytest.raises(OperationNotAllowedError, match='not allowed'):
        sut.execute_shell(f"echo 'Hello {random_string}!'", agent)

def test_execute_shell_allowlist_should_allow(agent: Agent, random_string: str):
    if False:
        for i in range(10):
            print('nop')
    agent.legacy_config.shell_command_control = sut.ALLOWLIST_CONTROL
    agent.legacy_config.shell_allowlist = ['echo']
    result = sut.execute_shell(f"echo 'Hello {random_string}!'", agent)
    assert 'Hello' in result and random_string in result