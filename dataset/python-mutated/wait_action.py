from tribler_apptester.action import Action

class WaitAction(Action):
    """
    This action simply waits (non-blocking) for a defined amount of time (in milliseconds).
    """

    def __init__(self, wait_time):
        if False:
            print('Hello World!')
        super(WaitAction, self).__init__()
        self.wait_time = wait_time

    def action_code(self):
        if False:
            while True:
                i = 10
        return 'QTest.qWait(%d)' % self.wait_time

    def required_imports(self):
        if False:
            print('Hello World!')
        return ['from PyQt5.QtTest import QTest']