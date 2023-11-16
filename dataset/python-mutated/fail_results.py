"""

Requestor Node that fails to download the results.

"""
import mock
import time
from golemapp import main

def pull_package(self, content_hash, task_id, subtask_id, key_or_secret, success, error, async_=True, client_options=None, output_dir=None):
    if False:
        print('Hello World!')
    time.sleep(3)
    error('wrench in the gears')
with mock.patch('golem.task.result.resultmanager.EncryptedResultPackageManager.pull_package', pull_package):
    main()