from tribler_apptester.action import Action

class CustomAction(Action):
    """
    This action allows programmers to define their own actions with custom code.
    """

    def __init__(self, code):
        if False:
            print('Hello World!')
        super(CustomAction, self).__init__()
        self.code = code

    def action_code(self):
        if False:
            return 10
        return self.code