from __future__ import absolute_import
from st2common.runners.base_action import Action

class MyAction(Action):

    def run(self):
        if False:
            return 10
        k1 = 'xyz'
        k2 = 'abc'
        k3 = True
        return {'k1': k1, 'k2': k2, 'k3': k3}