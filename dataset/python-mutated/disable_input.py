from pyboy.plugins.base_plugin import PyBoyPlugin

class DisableInput(PyBoyPlugin):
    argv = [('--no-input', {'action': 'store_true', 'help': 'Disable all user-input (mostly for autonomous testing)'})]

    def handle_events(self, events):
        if False:
            for i in range(10):
                print('nop')
        return []

    def enabled(self):
        if False:
            print('Hello World!')
        return self.pyboy_argv.get('no_input')