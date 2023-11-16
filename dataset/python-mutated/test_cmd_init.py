import contextlib
import os
import re
import shutil
import subprocess
import pytest
from lightning.app.cli import cmd_init
from lightning.app.utilities.imports import _IS_MACOS, _IS_WINDOWS

def test_validate_init_name():
    if False:
        print('Hello World!')
    value = cmd_init._capture_valid_app_component_name('abc1-cde')
    assert value == 'abc1-cde'
    value = cmd_init._capture_valid_app_component_name('abc-cde')
    assert value == 'abc-cde'
    with pytest.raises(SystemExit) as e:
        value = cmd_init._capture_valid_app_component_name('abc-cde#')
    assert 'Error: your Lightning app name' in str(e.value)

@pytest.mark.skipif(_IS_WINDOWS or _IS_MACOS, reason='unsupported OS')
@pytest.mark.xfail(strict=False, reason='need app fast_dev_run to work via CLI')
def test_make_app_template():
    if False:
        i = 10
        return i + 15
    template_name = 'app-test-template'
    template_name_folder = re.sub('-', '_', template_name)
    template_dir = os.path.join(os.getcwd(), template_name)
    with contextlib.suppress(Exception):
        shutil.rmtree(template_dir)
    subprocess.check_output(f'lightning init app {template_name}', shell=True)
    env_output = subprocess.check_output('pip freeze', shell=True)
    assert template_name not in str(env_output)
    env_output = subprocess.check_output(f'cd {template_name} && pip install -r requirements.txt && pip install -e .', shell=True)
    env_output = subprocess.check_output('pip freeze', shell=True)
    assert template_name in str(env_output)
    app_dir = os.path.join(template_dir, f'{template_name_folder}/app.py')
    output = subprocess.check_output(f'lightning run app {app_dir} --fast_dev_run')
    with contextlib.suppress(Exception):
        shutil.rmtree(template_dir)

@pytest.mark.xfail(strict=False, reason='need component fast_dev_run to work via CLI')
def test_make_component_template():
    if False:
        while True:
            i = 10
    template_name = 'component-test-template'
    template_name_folder = re.sub('-', '_', template_name)
    template_dir = os.path.join(os.getcwd(), template_name)
    with contextlib.suppress(Exception):
        shutil.rmtree(template_dir)
    subprocess.check_output(f'lightning init component {template_name}', shell=True)
    env_output = subprocess.check_output('pip freeze', shell=True)
    assert template_name not in str(env_output)
    env_output = subprocess.check_output(f'cd {template_name} && pip install -r requirements.txt && pip install -e .', shell=True)
    env_output = subprocess.check_output('pip freeze', shell=True)
    assert template_name in str(env_output)
    app_dir = os.path.join(template_dir, f'{template_name_folder}/app.py')
    output = subprocess.check_output(f'lightning run app {app_dir} --fast_dev_run')
    with contextlib.suppress(Exception):
        shutil.rmtree(template_dir)