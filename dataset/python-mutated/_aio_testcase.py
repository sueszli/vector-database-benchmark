import asyncio
from unittest.mock import Mock
from azure.core.credentials import AccessToken
from devtools_testutils import AzureMgmtTestCase

class AzureMgmtAsyncTestCase(AzureMgmtTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(AzureMgmtAsyncTestCase, self).setUp()

    @property
    def event_loop(self):
        if False:
            while True:
                i = 10
        return asyncio.get_event_loop()

    def create_mgmt_aio_client(self, client, **kwargs):
        if False:
            while True:
                i = 10
        if self.is_live:
            from azure.identity.aio import DefaultAzureCredential
            credential = DefaultAzureCredential()
        else:
            credential = Mock(get_token=asyncio.coroutine(lambda _: AccessToken('fake-token', 0)))
        return client(credential=credential, subscription_id=self.settings.SUBSCRIPTION_ID)

    def to_list(self, ait):
        if False:
            return 10

        async def lst():
            result = []
            async for item in ait:
                result.append(item)
            return result
        return self.event_loop.run_until_complete(lst())