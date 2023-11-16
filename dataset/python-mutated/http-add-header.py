"""Add an HTTP header to each response."""

class AddHeader:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.num = 0

    def response(self, flow):
        if False:
            for i in range(10):
                print('nop')
        self.num = self.num + 1
        flow.response.headers['count'] = str(self.num)
addons = [AddHeader()]