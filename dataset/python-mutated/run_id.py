import itertools
import uuid as _uuid

def increment(task, params=None) -> str:
    if False:
        print('Hello World!')
    'Generate Run IDs that are increment numbers'
    ids = [run.run_id for run in task._run_stack]
    for i in itertools.count(1):
        i = str(i)
        if i not in ids:
            return i

def uuid(task, params=None) -> str:
    if False:
        i = 10
        return i + 15
    return _uuid.uuid4().hex