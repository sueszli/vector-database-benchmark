from st2common.runners.base_action import Action

class ObjectReturnAction(Action):

    def run(self):
        if False:
            return 10
        return {'a': 'b', 'c': {'d': 'e', 'f': 1, 'g': True}}