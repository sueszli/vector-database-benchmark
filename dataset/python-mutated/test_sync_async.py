from tests import generic_tasks
from prefect import flow

def test_async_flow_from_sync_flow():
    if False:
        for i in range(10):
            print('nop')

    @flow
    async def async_run():
        return generic_tasks.noop()

    @flow
    def run():
        if False:
            for i in range(10):
                print('nop')
        async_run()
    run()