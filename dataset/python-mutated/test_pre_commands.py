import pytest
from conda import plugins
from conda.plugins.types import CondaPreCommand

class PreCommandPlugin:

    def pre_command_action(self, command: str) -> int:
        if False:
            i = 10
            return i + 15
        pass

    @plugins.hookimpl
    def conda_pre_commands(self):
        if False:
            for i in range(10):
                print('nop')
        yield CondaPreCommand(name='custom-pre-command', action=self.pre_command_action, run_for={'install', 'create', 'info'})

@pytest.fixture()
def pre_command_plugin(mocker, plugin_manager):
    if False:
        for i in range(10):
            print('nop')
    mocker.patch.object(PreCommandPlugin, 'pre_command_action')
    pre_command_plugin = PreCommandPlugin()
    plugin_manager.register(pre_command_plugin)
    return pre_command_plugin

def test_pre_command_invoked(pre_command_plugin, conda_cli):
    if False:
        while True:
            i = 10
    '\n    Makes sure that we successfully invoked our "pre-command" action.\n    '
    conda_cli('info')
    assert len(pre_command_plugin.pre_command_action.mock_calls) == 1

def test_pre_command_not_invoked(pre_command_plugin, conda_cli):
    if False:
        i = 10
        return i + 15
    '\n    Makes sure that we successfully did not invoke our "pre-command" action.\n    '
    conda_cli('config')
    assert len(pre_command_plugin.pre_command_action.mock_calls) == 0

def test_pre_command_action_raises_exception(pre_command_plugin, conda_cli):
    if False:
        return 10
    "\n    When the plugin action fails or raises an exception, we want to make sure\n    that it bubbles up to the top and isn't caught anywhere. This will ensure that it\n    goes through our normal exception catching/reporting mechanism.\n    "
    exc_message = '💥'
    pre_command_plugin.pre_command_action.side_effect = [Exception(exc_message)]
    with pytest.raises(Exception, match=exc_message):
        conda_cli('info')
    assert len(pre_command_plugin.pre_command_action.mock_calls) == 1