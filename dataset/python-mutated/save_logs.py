import os
from metaflow.datastore import FlowDataStore
from metaflow.plugins import DATASTORES
from metaflow.util import Path
from . import TASK_LOG_SOURCE
from metaflow.tracing import cli_entrypoint
SMALL_FILE_LIMIT = 1024 * 1024

@cli_entrypoint('save_logs')
def save_logs():
    if False:
        while True:
            i = 10

    def _read_file(path):
        if False:
            print('Hello World!')
        with open(path, 'rb') as f:
            return f.read()
    pathspec = os.environ['MF_PATHSPEC']
    attempt = os.environ['MF_ATTEMPT']
    ds_type = os.environ['MF_DATASTORE']
    ds_root = os.environ.get('MF_DATASTORE_ROOT')
    paths = (os.environ['MFLOG_STDOUT'], os.environ['MFLOG_STDERR'])
    (flow_name, run_id, step_name, task_id) = pathspec.split('/')
    storage_impl = [d for d in DATASTORES if d.TYPE == ds_type][0]
    if ds_root is None:

        def print_clean(line, **kwargs):
            if False:
                print('Hello World!')
            pass
        ds_root = storage_impl.get_datastore_root_from_config(print_clean)
    flow_datastore = FlowDataStore(flow_name, None, storage_impl=storage_impl, ds_root=ds_root)
    task_datastore = flow_datastore.get_task_datastore(run_id, step_name, task_id, int(attempt), mode='w')
    try:
        streams = ('stdout', 'stderr')
        sizes = [(stream, path, os.path.getsize(path)) for (stream, path) in zip(streams, paths) if os.path.exists(path)]
        if max((size for (_, _, size) in sizes)) < SMALL_FILE_LIMIT:
            op = _read_file
        else:
            op = Path
        data = {stream: op(path) for (stream, path, _) in sizes}
        task_datastore.save_logs(TASK_LOG_SOURCE, data)
    except:
        pass
if __name__ == '__main__':
    save_logs()
    "\n    import sys\n    from metaflow.metaflow_profile import profile\n    d = {}\n    with profile('save_logs', stats_dict=d):\n        save_logs()\n    print('Save logs took %dms' % d['save_logs'], file=sys.stderr)\n    "