from pathlib import Path
from typing import Sequence
import pytest
import ruamel.yaml
from tests.conftest import make_semgrepconfig_file
from semgrep.app.project_config import ProjectConfig
CONFIG_TAGS = 'tags:\n- tag1\n- tag_key:tag_val\n'
CONFIG_TAGS_MONOREPO_1 = 'tags:\n- tag1\n- service:service-1\n'
CONFIG_TAGS_MONOREPO_2 = 'tags:\n- tag1\n- service:service-2\n'

def create_mock_dir(git_tmp_path, files: Sequence[str]) -> None:
    if False:
        while True:
            i = 10
    for f in files:
        out_file = git_tmp_path / f
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text('x = 1')

@pytest.mark.quick
def test_projectconfig__find_all_config_files_basic(git_tmp_path):
    if False:
        i = 10
        return i + 15
    dir_files = ['test.py', 'main.py', 'setup.py']
    create_mock_dir(git_tmp_path, dir_files)
    make_semgrepconfig_file(git_tmp_path, CONFIG_TAGS)
    config_files = ProjectConfig._find_all_config_files(git_tmp_path, git_tmp_path)
    assert config_files == [git_tmp_path / '.semgrepconfig']

@pytest.mark.quick
def test_projectconfig__find_all_config_files_monorepo(git_tmp_path):
    if False:
        for i in range(10):
            print('nop')
    dir_files = ['service1/main.py', 'service2/main.py']
    create_mock_dir(git_tmp_path, dir_files)
    service_1_dir = git_tmp_path / 'service1'
    service_2_dir = git_tmp_path / 'service2'
    make_semgrepconfig_file(git_tmp_path, CONFIG_TAGS)
    make_semgrepconfig_file(service_1_dir, CONFIG_TAGS_MONOREPO_1)
    make_semgrepconfig_file(service_2_dir, CONFIG_TAGS_MONOREPO_2)
    config_files = ProjectConfig._find_all_config_files(git_tmp_path, service_1_dir)
    assert git_tmp_path / '.semgrepconfig' in config_files
    assert service_1_dir / '.semgrepconfig' in config_files
    assert service_2_dir / '.semgrepconfig' not in config_files

@pytest.mark.quick
def test_projectconfig_load_all_basic(git_tmp_path, mocker):
    if False:
        print('Hello World!')
    dir_files = ['test.py', 'main.py', 'setup.py']
    create_mock_dir(git_tmp_path, dir_files)
    make_semgrepconfig_file(git_tmp_path, CONFIG_TAGS)
    mocker.patch.object(Path, 'cwd', return_value=git_tmp_path)
    mocker.patch('semgrep.git.get_git_root_path', return_value=git_tmp_path)
    proj_config = ProjectConfig.load_all()
    expected_tags = ['tag1', 'tag_key:tag_val']
    assert proj_config.tags == expected_tags

@pytest.mark.quick
def test_projectconfig_load_all_monorepo(git_tmp_path, mocker):
    if False:
        return 10
    dir_files = ['service1/main.py', 'service2/main.py']
    create_mock_dir(git_tmp_path, dir_files)
    service_1_dir = git_tmp_path / 'service1'
    service_2_dir = git_tmp_path / 'service2'
    make_semgrepconfig_file(git_tmp_path, CONFIG_TAGS)
    make_semgrepconfig_file(service_1_dir, CONFIG_TAGS_MONOREPO_1)
    make_semgrepconfig_file(service_2_dir, CONFIG_TAGS_MONOREPO_2)
    mocker.patch.object(Path, 'cwd', return_value=service_1_dir)
    mocker.patch('semgrep.git.get_git_root_path', return_value=git_tmp_path)
    proj_config = ProjectConfig.load_all()
    expected_tags = ['tag1', 'service:service-1']
    assert proj_config.tags == expected_tags

@pytest.mark.quick
def test_projectconfig_load_from_file_invalid_format(tmp_path):
    if False:
        while True:
            i = 10
    tmp_file = tmp_path / '.semgrepconfig'
    yaml = ruamel.yaml.YAML(typ='safe')
    invalid_cfg = {'version': 'v1', 'tags': {'tag1': 'value1'}}
    with tmp_file.open('w') as f:
        yaml.dump(invalid_cfg, f)
    with pytest.raises(ValueError):
        ProjectConfig.load_from_file(tmp_file)

@pytest.mark.quick
def test_projectconfig_todict():
    if False:
        while True:
            i = 10
    project_config = ProjectConfig(version='v1', tags=['tag1', 'tag2'])
    expected = {'version': 'v1', 'tags': ['tag1', 'tag2']}
    assert project_config.to_CiConfigFromRepo().to_json() == expected