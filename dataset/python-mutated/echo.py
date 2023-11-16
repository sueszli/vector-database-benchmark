from lwe.core.plugin import Plugin
import lwe.core.util as util

class Echo(Plugin):
    """
    Simple echo plugin, echos back the text you give it
    """

    def default_config(self):
        if False:
            i = 10
            return i + 15
        "\n        The default configuration for this plugin.\n        This is called by the plugin manager after the plugin is initialized.\n        The user can override these settings in their profile configuration,\n        under the key 'plugins.echo'.\n        "
        return {'response': {'prefix': 'Echo'}}

    def setup(self):
        if False:
            i = 10
            return i + 15
        '\n        Setup the plugin. This is called by the plugin manager after the backend\n        is initialized.\n        '
        self.log.info(f'This is the echo plugin, running with backend: {self.backend.name}')
        self.response_prefix = self.config.get('plugins.echo.response.prefix')

    def get_shell_completions(self, _base_shell_completions):
        if False:
            return 10
        'Example of provided shell completions.'
        commands = {}
        commands[util.command_with_leader('echo')] = util.list_to_completion_hash(['one', 'two', 'three'])
        return commands

    def command_echo(self, arg):
        if False:
            i = 10
            return i + 15
        '\n        Echo command, a simple plugin example\n\n        This command is provided as an example of extending functionality via a plugin.\n\n        Arguments:\n            text: The text to echo\n\n        Examples:\n            {COMMAND} one\n        '
        if not arg:
            return (False, arg, 'Argument is required')
        return (True, arg, f'{self.response_prefix}: {arg}')