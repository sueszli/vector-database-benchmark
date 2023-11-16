from azure.core.credentials import AccessToken

class FakeTokenCredential(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.token = AccessToken('Fake Token', 0)

    def get_token(self, *args, **kwargs):
        if False:
            print('Hello World!')
        return self.token