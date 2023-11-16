from tribler_apptester.action import Action

class ActionSequence(Action):
    """
    An action sequence is a list of actions. This allows programmers to define their own complicated sequence.
    """

    def __init__(self, actions=None):
        if False:
            return 10
        super().__init__()
        self.actions = actions or []

    def add_action(self, action):
        if False:
            for i in range(10):
                print('nop')
        self.actions.append(action)

    def get_required_imports(self):
        if False:
            i = 10
            return i + 15
        result = set(self.required_imports())
        for action in self.actions:
            result.update(action.get_required_imports())
        return sorted(result)

    def action_code(self):
        if False:
            for i in range(10):
                print('nop')
        return '\n'.join((action.action_code() for action in self.actions))