"""Tests that restart the cluster. Isolated from other event tests."""
import asyncio
import subprocess
import time
import pytest
import ray
from ray import workflow
from ray.tests.conftest import *
from ray.workflow.tests import utils

def test_crash_during_event_checkpointing(workflow_start_regular):
    if False:
        i = 10
        return i + 15
    'Ensure that if the cluster dies while the event is being checkpointed, we\n    properly re-poll for the event.'
    from ray._private import storage
    from ray.workflow.tests.utils import skip_client_mode_test
    skip_client_mode_test()
    storage_uri = storage._storage_uri
    "Ensure that we don't re-call poll_for_event after `event_checkpointed`\n       returns, even after a crash."

    class MyEventListener(workflow.EventListener):

        async def poll_for_event(self):
            assert not utils.check_global_mark('committed')
            if utils.check_global_mark('first'):
                utils.set_global_mark('second')
            utils.set_global_mark('first')
            utils.set_global_mark('time_to_die')
            while not utils.check_global_mark('resume'):
                time.sleep(0.1)

        async def event_checkpointed(self, event):
            utils.set_global_mark('committed')

    @ray.remote
    def wait_then_finish(arg):
        if False:
            return 10
        pass
    event_promise = workflow.wait_for_event(MyEventListener)
    workflow.run_async(wait_then_finish.bind(event_promise), workflow_id='workflow')
    while not utils.check_global_mark('time_to_die'):
        time.sleep(0.1)
    assert utils.check_global_mark('first')
    ray.shutdown()
    subprocess.check_output(['ray', 'stop', '--force'])
    ray.init(num_cpus=4, storage=storage_uri)
    workflow.init()
    workflow.resume_async('workflow')
    utils.set_global_mark('resume')
    workflow.get_output('workflow')
    assert utils.check_global_mark('second')

@pytest.mark.parametrize('workflow_start_regular', [{'num_cpus': 4}], indirect=True)
def test_crash_after_commit(workflow_start_regular):
    if False:
        while True:
            i = 10
    "Ensure that we don't re-call poll_for_event after `event_checkpointed`\n    returns, even after a crash. Here we must call `event_checkpointed`\n    twice, because there's no way to know if we called it after\n    checkpointing.\n    "
    from ray._private import storage
    from ray.workflow.tests.utils import skip_client_mode_test
    skip_client_mode_test()
    storage_uri = storage._storage_uri

    class MyEventListener(workflow.EventListener):

        async def poll_for_event(self):
            assert not utils.check_global_mark('committed')

        async def event_checkpointed(self, event):
            utils.set_global_mark('committed')
            if utils.check_global_mark('first'):
                utils.set_global_mark('second')
            else:
                utils.set_global_mark('first')
                await asyncio.sleep(1000000)
    event_promise = workflow.wait_for_event(MyEventListener)
    workflow.run_async(event_promise, workflow_id='workflow')
    while not utils.check_global_mark('first'):
        time.sleep(0.1)
    ray.shutdown()
    subprocess.check_output(['ray', 'stop', '--force'])
    ray.init(num_cpus=4, storage=storage_uri)
    workflow.init()
    workflow.resume_async('workflow')
    workflow.get_output('workflow')
    assert utils.check_global_mark('second')
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))