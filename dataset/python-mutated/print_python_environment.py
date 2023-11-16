import os
import sys
import platform
from st2common.runners.base_action import Action

class PrintPythonEnvironmentAction(Action):

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        print('Using Python executable: %s' % sys.executable)
        print('Using Python version: %s' % sys.version)
        print('Platform: %s' % platform.platform())
        print('PYTHONPATH: %s' % os.environ.get('PYTHONPATH'))
        print('sys.path: %s' % sys.path)