from azure.core.credentials import AccessToken

class AsyncFakeTokenCredential(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.token = AccessToken('Fake Token', 0)

    async def get_token(self, *args, **kwargs):
        return self.token