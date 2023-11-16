from __future__ import absolute_import
from st2common.runners.base_action import Action

class PyAction(Action):

    def run(self, k1):
        if False:
            i = 10
            return i + 15
        return {'k2': k1}