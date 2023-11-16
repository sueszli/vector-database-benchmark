"""
.. module: security_monkey.tests.core.mock_monitor
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Bridgewater OSS <opensource@bwater.com>

"""
from security_monkey.watcher import ChangeItem
from collections import defaultdict
RUNTIME_WATCHERS = defaultdict(list)
RUNTIME_AUDIT_COUNTS = defaultdict(list)
CURRENT_MONITORS = []

class MockMonitor(object):

    def __init__(self, watcher, auditors):
        if False:
            for i in range(10):
                print('nop')
        self.watcher = watcher
        self.auditors = auditors
        self.batch_support = self.watcher.batched_size > 0

class MockRunnableWatcher(object):

    def __init__(self, index, interval):
        if False:
            return 10
        self.index = index
        self.interval = interval
        self.i_am_singular = index
        self.created_items = []
        self.deleted_items = []
        self.changed_items = []
        self.batched_size = 0
        self.done_slurping = True
        self.total_list = []
        self.batch_counter = 0

    def slurp(self):
        if False:
            while True:
                i = 10
        RUNTIME_WATCHERS[self.index].append(self)
        item_list = []
        exception_map = {}
        return (item_list, exception_map)

    def save(self):
        if False:
            print('Hello World!')
        pass

    def get_interval(self):
        if False:
            print('Hello World!')
        return self.interval

    def find_changes(self, current=[], exception_map={}):
        if False:
            print('Hello World!')
        self.created_items.append(ChangeItem(index=self.index))

class MockRunnableAuditor(object):

    def __init__(self, index, support_auditor_indexes, support_watcher_indexes):
        if False:
            for i in range(10):
                print('nop')
        self.index = index
        self.support_auditor_indexes = support_auditor_indexes
        self.support_watcher_indexes = support_watcher_indexes
        self.items = []

    def audit_objects(self):
        if False:
            for i in range(10):
                print('nop')
        item_count = RUNTIME_AUDIT_COUNTS.get(self.index, 0)
        RUNTIME_AUDIT_COUNTS[self.index] = item_count + len(self.items)

    def save_issues(self):
        if False:
            i = 10
            return i + 15
        pass

    def applies_to_account(self, db_account):
        if False:
            return 10
        return True

    def read_previous_items(self):
        if False:
            return 10
        return [ChangeItem(index=self.index)]

def build_mock_result(watcher_configs, auditor_configs):
    if False:
        i = 10
        return i + 15
    '\n    Builds mock monitor results that can be used to override the results of the\n    monitor methods.\n    '
    del CURRENT_MONITORS[:]
    for config in watcher_configs:
        watcher = mock_watcher(config)
        auditors = []
        for config in auditor_configs:
            if config['index'] == watcher.index:
                auditors.append(mock_auditor(config))
        CURRENT_MONITORS.append(MockMonitor(watcher, auditors))

def mock_watcher(config):
    if False:
        print('Hello World!')
    "\n    Builds a mock watcher from a config dictionary like:\n    {\n        'index': 'index1',\n        'interval: 15'\n    }\n    "
    return MockRunnableWatcher(config['index'], config['interval'])

def mock_auditor(config):
    if False:
        return 10
    "\n    Builds a mock auditor from a config dictionary like:\n    {\n        'index': 'index1',\n        'support_auditor_indexes': [],\n        'support_watcher_indexes': ['index2']\n    }\n    "
    return MockRunnableAuditor(config['index'], config['support_auditor_indexes'], config['support_watcher_indexes'])

def mock_all_monitors(account_name, debug=False):
    if False:
        return 10
    return CURRENT_MONITORS

def mock_get_monitors(account_name, monitor_names, debug=False):
    if False:
        for i in range(10):
            print('nop')
    monitors = []
    for monitor in CURRENT_MONITORS:
        if monitor.watcher.index in monitor_names:
            monitors.append(monitor)
    return monitors