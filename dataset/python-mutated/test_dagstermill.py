import os
from contextlib import contextmanager
from dagstermill_tests.test_ops import cleanup_result_notebook
from dagster import execute_job
from dagster._core.definitions.reconstruct import ReconstructableJob
from dagster._core.test_utils import instance_for_test
IS_BUILDKITE = os.getenv('BUILDKITE') is not None
import subprocess
import warnings
import pytest

@pytest.fixture(autouse=True)
def kernel():
    if False:
        return 10
    warnings.warn("Installing Jupyter kernel dagster. Don't worry, this is noninvasive and you can reverse it by running `jupyter kernelspec uninstall dagster`.")
    subprocess.check_output(['ipython', 'kernel', 'install', '--name', 'dagster', '--user'])

@contextmanager
def exec_for_test(module_name, fn_name, env=None, raise_on_error=True, **kwargs):
    if False:
        print('Hello World!')
    result = None
    recon_job = ReconstructableJob.for_module(module_name, fn_name)
    with instance_for_test() as instance:
        try:
            with execute_job(recon_job, run_config=env, instance=instance, raise_on_error=raise_on_error, **kwargs) as result:
                yield result
        finally:
            if result:
                cleanup_result_notebook(result)

@pytest.mark.flaky(reruns=1)
def test_config_asset():
    if False:
        return 10
    module_path = 'docs_snippets.integrations.dagstermill.iris_notebook_config'
    if not IS_BUILDKITE:
        module_path = 'examples.docs_snippets.' + module_path
    with exec_for_test(module_name=module_path, fn_name='config_asset_job') as result:
        assert result.success

@pytest.mark.flaky(reruns=1)
def test_iris_classify_job():
    if False:
        for i in range(10):
            print('nop')
    module_path = 'docs_snippets.integrations.dagstermill.iris_notebook_op'
    if not IS_BUILDKITE:
        module_path = 'examples.docs_snippets.' + module_path
    with exec_for_test(module_name=module_path, fn_name='iris_classify') as result:
        assert result.success

@pytest.mark.flaky(reruns=1)
def test_outputs_job():
    if False:
        i = 10
        return i + 15
    module_path = 'docs_snippets.integrations.dagstermill.notebook_outputs'
    if not IS_BUILDKITE:
        module_path = 'examples.docs_snippets.' + module_path
    with exec_for_test(module_name=module_path, fn_name='my_job') as result:
        assert result.success