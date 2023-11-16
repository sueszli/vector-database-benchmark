import os
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from typing import Any, Iterator, Mapping, Optional
import yaml
from dagster._utils.error import serializable_error_info_from_exc_info
from .._utils.env import environ
from .._utils.merger import merge_dicts
from .instance import DagsterInstance

@contextmanager
def instance_for_test(overrides: Optional[Mapping[str, Any]]=None, set_dagster_home: bool=True, temp_dir: Optional[str]=None) -> Iterator[DagsterInstance]:
    if False:
        while True:
            i = 10
    "Creates a persistent :py:class:`~dagster.DagsterInstance` available within a context manager.\n\n    When a context manager is opened, if no `temp_dir` parameter is set, a new\n    temporary directory will be created for the duration of the context\n    manager's opening. If the `set_dagster_home` parameter is set to True\n    (True by default), the `$DAGSTER_HOME` environment variable will be\n    overridden to be this directory (or the directory passed in by `temp_dir`)\n    for the duration of the context manager being open.\n\n    Args:\n        overrides (Optional[Mapping[str, Any]]):\n            Config to provide to instance (config format follows that typically found in an `instance.yaml` file).\n        set_dagster_home (Optional[bool]):\n            If set to True, the `$DAGSTER_HOME` environment variable will be\n            overridden to be the directory used by this instance for the\n            duration that the context manager is open. Upon the context\n            manager closing, the `$DAGSTER_HOME` variable will be re-set to the original value. (Defaults to True).\n        temp_dir (Optional[str]):\n            The directory to use for storing local artifacts produced by the\n            instance. If not set, a temporary directory will be created for\n            the duration of the context manager being open, and all artifacts\n            will be torn down afterward.\n    "
    with ExitStack() as stack:
        if not temp_dir:
            temp_dir = stack.enter_context(tempfile.TemporaryDirectory())
        instance_overrides = merge_dicts({'telemetry': {'enabled': False}, 'code_servers': {'wait_for_local_processes_on_shutdown': True}}, overrides if overrides else {})
        if set_dagster_home:
            stack.enter_context(environ({'DAGSTER_HOME': temp_dir, 'DAGSTER_DISABLE_TELEMETRY': 'yes'}))
        with open(os.path.join(temp_dir, 'dagster.yaml'), 'w', encoding='utf8') as fd:
            yaml.dump(instance_overrides, fd, default_flow_style=False)
        with DagsterInstance.from_config(temp_dir) as instance:
            try:
                yield instance
            except:
                sys.stderr.write('Test raised an exception, attempting to clean up instance:' + serializable_error_info_from_exc_info(sys.exc_info()).to_string() + '\n')
                raise
            finally:
                cleanup_test_instance(instance)

def cleanup_test_instance(instance: DagsterInstance) -> None:
    if False:
        print('Hello World!')
    if instance._run_launcher:
        instance._run_launcher.join()