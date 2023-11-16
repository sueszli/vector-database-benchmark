from tribler_apptester.action import Action

class KeyAction(Action):
    """
    This action presses a specific keyboard key.
    """

    def __init__(self, obj_name, key_name):
        if False:
            i = 10
            return i + 15
        super(KeyAction, self).__init__()
        self.obj_name = obj_name
        self.key_name = key_name

    def action_code(self):
        if False:
            for i in range(10):
                print('nop')
        return 'QTest.keyClick(%s, Qt.%s)' % (self.obj_name, self.key_name)

    def required_imports(self):
        if False:
            i = 10
            return i + 15
        return ['from PyQt5.QtTest import QTest', 'from PyQt5.QtCore import Qt']