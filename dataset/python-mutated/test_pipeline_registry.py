import sys
import textwrap
import pytest
from kedro.framework.project import configure_project, pipelines
from kedro.pipeline import Pipeline

@pytest.fixture
def mock_package_name_with_pipelines_file(tmpdir):
    if False:
        print('Hello World!')
    pipelines_file_path = tmpdir.mkdir('test_package') / 'pipeline_registry.py'
    pipelines_file_path.write(textwrap.dedent('\n                from kedro.pipeline import Pipeline\n                def register_pipelines():\n                    return {"new_pipeline": Pipeline([])}\n            '))
    (project_path, package_name, _) = str(pipelines_file_path).rpartition('test_package')
    sys.path.insert(0, project_path)
    yield package_name
    sys.path.pop(0)

def test_pipelines_without_configure_project_is_empty(mock_package_name_with_pipelines_file):
    if False:
        for i in range(10):
            print('nop')
    del sys.modules['kedro.framework.project']
    from kedro.framework.project import pipelines
    assert pipelines == {}

@pytest.fixture
def mock_package_name_with_unimportable_pipelines_file(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    pipelines_file_path = tmpdir.mkdir('test_broken_package') / 'pipeline_registry.py'
    pipelines_file_path.write(textwrap.dedent('\n                import this_is_not_a_real_thing\n                from kedro.pipeline import Pipeline\n                def register_pipelines():\n                    return {"new_pipeline": Pipeline([])}\n            '))
    (project_path, package_name, _) = str(pipelines_file_path).rpartition('test_broken_package')
    sys.path.insert(0, project_path)
    yield package_name
    sys.path.pop(0)

def test_pipelines_after_configuring_project_shows_updated_values(mock_package_name_with_pipelines_file):
    if False:
        return 10
    configure_project(mock_package_name_with_pipelines_file)
    assert isinstance(pipelines['new_pipeline'], Pipeline)

def test_configure_project_should_not_raise_for_unimportable_pipelines(mock_package_name_with_unimportable_pipelines_file):
    if False:
        return 10
    configure_project(mock_package_name_with_unimportable_pipelines_file)
    with pytest.raises(ModuleNotFoundError, match="No module named 'this_is_not_a_real_thing'"):
        _ = pipelines['new_pipeline']