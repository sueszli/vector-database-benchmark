from fail2ban.server.action import ActionBase

class TestAction(ActionBase):

    def __init__(self, jail, name):
        if False:
            print('Hello World!')
        super(TestAction, self).__init__(jail, name)

    def start(self):
        if False:
            while True:
                i = 10
        raise Exception()

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        raise Exception()

    def ban(self):
        if False:
            for i in range(10):
                print('nop')
        raise Exception()

    def unban(self):
        if False:
            while True:
                i = 10
        raise Exception()
Action = TestAction