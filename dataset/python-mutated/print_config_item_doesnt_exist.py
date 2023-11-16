from __future__ import absolute_import
from st2common.runners.base_action import Action

class PrintConfigItemAction(Action):

    def run(self):
        if False:
            print('Hello World!')
        print(self.config)
        print(self.config.get('item1', 'default_value'))
        print(self.config['key'])