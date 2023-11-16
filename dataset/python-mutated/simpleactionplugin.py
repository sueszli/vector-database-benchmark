"""Sample implementation for ActionMixin."""
from plugin import InvenTreePlugin
from plugin.mixins import ActionMixin

class SimpleActionPlugin(ActionMixin, InvenTreePlugin):
    """An EXTREMELY simple action plugin which demonstrates the capability of the ActionMixin class."""
    NAME = 'SimpleActionPlugin'
    ACTION_NAME = 'simple'

    def perform_action(self, user=None, data=None):
        if False:
            for i in range(10):
                print('nop')
        'Sample method.'
        print('Action plugin in action!')

    def get_info(self, user, data=None):
        if False:
            while True:
                i = 10
        'Sample method.'
        return {'user': user.username, 'hello': 'world'}

    def get_result(self, user=None, data=None):
        if False:
            print('Hello World!')
        'Sample method.'
        return True