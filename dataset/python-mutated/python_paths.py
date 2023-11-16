from __future__ import absolute_import
import os
import sys
from st2common.runners.base_action import Action

class PythonPathsAction(Action):

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        print('sys.path: %s' % sys.path)
        print('PYTHONPATH: %s' % os.environ.get('PYTHONPATH'))