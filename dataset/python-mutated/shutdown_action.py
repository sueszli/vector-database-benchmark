from tribler_apptester.action import Action

class ShutdownAction(Action):
    """
    This action shuts down Tribler.
    """

    def action_code(self):
        if False:
            for i in range(10):
                print('nop')
        return 'window.close_tribler()'

class HardShutdownAction(Action):
    """
    This action shuts down Tribler in a more forced way.
    """

    def action_code(self):
        if False:
            for i in range(10):
                print('nop')
        return 'QApplication.quit()'

    def required_imports(self):
        if False:
            i = 10
            return i + 15
        return ['from PyQt5.QtWidgets import QApplication']