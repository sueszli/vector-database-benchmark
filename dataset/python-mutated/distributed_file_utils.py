"""Utilities that help manage directory path in distributed settings.

In multi-worker training, the need to write a file to distributed file
location often requires only one copy done by one worker despite many workers
that are involved in training. The option to only perform saving by chief is
not feasible for a couple of reasons: 1) Chief and workers may each contain
a client that runs the same piece of code and it's preferred not to make
any distinction between the code run by chief and other workers, and 2)
saving of model or model's related information may require SyncOnRead
variables to be read, which needs the cooperation of all workers to perform
all-reduce.

This set of utility is used so that only one copy is written to the needed
directory, by supplying a temporary write directory path for workers that don't
need to save, and removing the temporary directory once file writing is done.

Example usage:
```
# Before using a directory to write file to.
self.log_write_dir = write_dirpath(self.log_dir, get_distribution_strategy())
# Now `self.log_write_dir` can be safely used to write file to.

...

# After the file is written to the directory.
remove_temp_dirpath(self.log_dir, get_distribution_strategy())

```

Experimental. API is subject to change.
"""
import os
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.lib.io import file_io

def _get_base_dirpath(strategy):
    if False:
        while True:
            i = 10
    task_id = strategy.extended._task_id
    return 'workertemp_' + str(task_id)

def _is_temp_dir(dirpath, strategy):
    if False:
        print('Hello World!')
    return dirpath.endswith(_get_base_dirpath(strategy))

def _get_temp_dir(dirpath, strategy):
    if False:
        for i in range(10):
            print('nop')
    if _is_temp_dir(dirpath, strategy):
        temp_dir = dirpath
    else:
        temp_dir = os.path.join(dirpath, _get_base_dirpath(strategy))
    file_io.recursive_create_dir_v2(temp_dir)
    return temp_dir

def write_dirpath(dirpath, strategy):
    if False:
        i = 10
        return i + 15
    "Returns the writing dir that should be used to save file distributedly.\n\n  `dirpath` would be created if it doesn't exist.\n\n  Args:\n    dirpath: Original dirpath that would be used without distribution.\n    strategy: The tf.distribute strategy object currently used.\n\n  Returns:\n    The writing dir path that should be used to save with distribution.\n  "
    if strategy is None:
        strategy = distribute_lib.get_strategy()
    if strategy is None:
        return dirpath
    if not strategy.extended._in_multi_worker_mode():
        return dirpath
    if strategy.extended.should_checkpoint:
        return dirpath
    return _get_temp_dir(dirpath, strategy)

def remove_temp_dirpath(dirpath, strategy):
    if False:
        return 10
    'Removes the temp path after writing is finished.\n\n  Args:\n    dirpath: Original dirpath that would be used without distribution.\n    strategy: The tf.distribute strategy object currently used.\n  '
    if strategy is None:
        strategy = distribute_lib.get_strategy()
    if strategy is None:
        return
    if strategy.extended._in_multi_worker_mode() and (not strategy.extended.should_checkpoint):
        file_io.delete_recursively(_get_temp_dir(dirpath, strategy))

def write_filepath(filepath, strategy):
    if False:
        return 10
    "Returns the writing file path to be used to save file distributedly.\n\n  Directory to contain `filepath` would be created if it doesn't exist.\n\n  Args:\n    filepath: Original filepath that would be used without distribution.\n    strategy: The tf.distribute strategy object currently used.\n\n  Returns:\n    The writing filepath that should be used to save file with distribution.\n  "
    dirpath = os.path.dirname(filepath)
    base = os.path.basename(filepath)
    return os.path.join(write_dirpath(dirpath, strategy), base)

def remove_temp_dir_with_filepath(filepath, strategy):
    if False:
        return 10
    'Removes the temp path for file after writing is finished.\n\n  Args:\n    filepath: Original filepath that would be used without distribution.\n    strategy: The tf.distribute strategy object currently used.\n  '
    remove_temp_dirpath(os.path.dirname(filepath), strategy)