import asyncio
import functools
from devtools_testutils import AzureRecordedTestCase
from azure_devtools.scenario_tests.utilities import trim_kwargs_from_test_function
from azure.agrifood.farming.aio import FarmBeatsClient

class FarmBeatsAsyncTestCase(AzureRecordedTestCase):

    def create_client(self, agrifood_endpoint) -> FarmBeatsClient:
        if False:
            print('Hello World!')
        self.credential = self.get_credential(FarmBeatsClient, is_async=True)
        self.client = self.create_client_from_credential(FarmBeatsClient, endpoint=agrifood_endpoint, credential=self.credential)
        return self.client

    async def close_client(self):
        await self.credential.close()
        await self.client.close()

    @staticmethod
    def await_prepared_test(test_fn):
        if False:
            i = 10
            return i + 15
        "Synchronous wrapper for async test methods. Used to avoid making changes\n        upstream to AbstractPreparer (which doesn't await the functions it wraps)\n        "

        @functools.wraps(test_fn)
        def run(test_class_instance, *args, **kwargs):
            if False:
                print('Hello World!')
            trim_kwargs_from_test_function(test_fn, kwargs)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            asyncio.run(test_fn(test_class_instance, **kwargs))
        return run