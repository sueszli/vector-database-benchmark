from __future__ import annotations
import os
from pathlib import Path
from unittest import mock
from airflow_breeze.utils.path_utils import find_airflow_sources_root_to_operate_on
ACTUAL_AIRFLOW_SOURCES = Path(__file__).parents[3].resolve()
ROOT_PATH = Path(Path(__file__).root)

def test_find_airflow_root_upwards_from_cwd(capsys):
    if False:
        while True:
            i = 10
    os.chdir(Path(__file__).parent)
    sources = find_airflow_sources_root_to_operate_on()
    assert sources == ACTUAL_AIRFLOW_SOURCES
    output = str(capsys.readouterr().out)
    assert output == ''

def test_find_airflow_root_upwards_from_file(capsys):
    if False:
        while True:
            i = 10
    os.chdir(Path(__file__).root)
    sources = find_airflow_sources_root_to_operate_on()
    assert sources == ACTUAL_AIRFLOW_SOURCES
    output = str(capsys.readouterr().out)
    assert output == ''

@mock.patch('airflow_breeze.utils.path_utils.AIRFLOW_CFG_FILE', 'bad_name.cfg')
@mock.patch('airflow_breeze.utils.path_utils.Path.cwd')
def test_find_airflow_root_from_installation_dir(mock_cwd, capsys):
    if False:
        while True:
            i = 10
    mock_cwd.return_value = ROOT_PATH
    sources = find_airflow_sources_root_to_operate_on()
    assert sources == ACTUAL_AIRFLOW_SOURCES