""" Define Pytest plugins for Jupyter Notebook tests.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
import os
import subprocess
import sys
import time
from os.path import dirname, exists, join, pardir
from typing import IO, Any, Callable
import pytest
import requests
from requests.exceptions import ConnectionError
from bokeh.util.terminal import write
pytest_plugins = ('tests.support.plugins.log_file',)
__all__ = ('jupyter_notebook',)

@pytest.fixture(scope='session')
def jupyter_notebook(request: pytest.FixtureRequest, log_file: IO[str]) -> str:
    if False:
        i = 10
        return i + 15
    '\n    Starts a jupyter notebook server at the beginning of a session, and\n    closes at the end of a session.\n\n    Adds custom.js that runs all the cells on notebook opening. Cleans out\n    this custom.js at the end of the test run.\n\n    Returns the url that the jupyter notebook is running at.\n\n    '
    from jupyter_core import paths
    config_dir = paths.jupyter_config_dir()
    body = '\nrequire(["base/js/namespace", "base/js/events"], function (IPython, events) {\n    events.on("kernel_ready.Kernel", function () {\n        IPython.notebook.execute_all_cells();\n    });\n});\n'
    custom = join(config_dir, 'custom')
    if not exists(custom):
        os.makedirs(custom)
    customjs = join(custom, 'custom.js')
    old_customjs = None
    if exists(customjs):
        with open(customjs) as f:
            old_customjs = f.read()
    with open(customjs, 'w') as f:
        f.write(body)

    def clean_up_customjs() -> None:
        if False:
            print('Hello World!')
        text = old_customjs if old_customjs is not None else ''
        with open(customjs, 'w') as f:
            f.write(text)
    request.addfinalizer(clean_up_customjs)
    notebook_port = request.config.option.notebook_port
    env = os.environ.copy()
    env['BOKEH_RESOURCES'] = 'server'
    notebook_dir = join(dirname(__file__), pardir, pardir)
    cmd = ['jupyter', 'notebook']
    argv = ['--no-browser', f'--port={notebook_port}', f'--notebook-dir={notebook_dir}']
    jupter_notebook_url = f'http://localhost:{notebook_port}'
    try:
        proc = subprocess.Popen(cmd + argv, env=env, stdout=log_file, stderr=log_file)
    except OSError:
        write(f"Failed to run: {' '.join(cmd + argv)}")
        sys.exit(1)
    else:

        def stop_jupyter_notebook() -> None:
            if False:
                print('Hello World!')
            write('Shutting down jupyter-notebook ...')
            proc.kill()
        request.addfinalizer(stop_jupyter_notebook)

        def wait_until(func: Callable[[], Any], timeout: float=5.0, interval: float=0.01) -> bool:
            if False:
                print('Hello World!')
            start = time.time()
            while True:
                if func():
                    return True
                if time.time() - start > timeout:
                    return False
                time.sleep(interval)

        def wait_for_jupyter_notebook() -> bool:
            if False:
                return 10

            def helper() -> Any:
                if False:
                    i = 10
                    return i + 15
                if proc.returncode is not None:
                    return True
                try:
                    return requests.get(jupter_notebook_url)
                except ConnectionError:
                    return False
            return wait_until(helper)
        if not wait_for_jupyter_notebook():
            write(f"Timeout when running: {' '.join(cmd + argv)}")
            sys.exit(1)
        if proc.returncode is not None:
            write(f'Jupyter notebook exited with code {proc.returncode}')
            sys.exit(1)
        return jupter_notebook_url