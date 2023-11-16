import sys
from st2common.runners.base_action import Action

class PrintPythonVersionAction(Action):

    def run(self):
        if False:
            return 10
        print('Using Python executable: %s' % sys.executable)
        print('Using Python version: %s' % sys.version)