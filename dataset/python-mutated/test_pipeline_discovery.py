import shutil
import sys
import textwrap
import warnings
from pathlib import Path
import pytest
from kedro.framework.project import configure_project, find_pipelines

@pytest.fixture
def mock_package_name_with_pipelines(tmp_path, request):
    if False:
        print('Hello World!')
    package_name = 'test_package'
    pipelines_dir = tmp_path / package_name / 'pipelines'
    pipelines_dir.mkdir(parents=True)
    (pipelines_dir / '__init__.py').touch()
    for pipeline_name in request.param:
        pipeline_dir = pipelines_dir / pipeline_name
        pipeline_dir.mkdir()
        (pipeline_dir / '__init__.py').write_text(textwrap.dedent(f'\n                from kedro.pipeline import Pipeline, node, pipeline\n\n\n                def create_pipeline(**kwargs) -> Pipeline:\n                    return pipeline([node(lambda: 1, None, "{pipeline_name}")])\n                '))
    sys.path.insert(0, str(tmp_path))
    yield package_name
    sys.path.pop(0)
    if f'{package_name}.pipeline' in sys.modules:
        del sys.modules[f'{package_name}.pipeline']
    if f'{package_name}.pipelines' in sys.modules:
        del sys.modules[f'{package_name}.pipelines']

@pytest.fixture
def pipeline_names(request):
    if False:
        return 10
    return request.param

@pytest.mark.parametrize('mock_package_name_with_pipelines,pipeline_names', [(x, x) for x in [set(), {'my_pipeline'}]], indirect=True)
def test_find_pipelines(mock_package_name_with_pipelines, pipeline_names):
    if False:
        i = 10
        return i + 15
    configure_project(mock_package_name_with_pipelines)
    pipelines = find_pipelines()
    assert set(pipelines) == pipeline_names | {'__default__'}
    assert sum(pipelines.values()).outputs() == pipeline_names

@pytest.mark.parametrize('mock_package_name_with_pipelines,pipeline_names', [(x, x) for x in [set(), {'good_pipeline'}]], indirect=True)
def test_find_pipelines_skips_modules_without_create_pipelines_function(mock_package_name_with_pipelines, pipeline_names):
    if False:
        i = 10
        return i + 15
    pipelines_dir = Path(sys.path[0]) / mock_package_name_with_pipelines / 'pipelines'
    pipeline_dir = pipelines_dir / 'bad_touch'
    pipeline_dir.mkdir()
    (pipeline_dir / '__init__.py').touch()
    configure_project(mock_package_name_with_pipelines)
    with pytest.warns(UserWarning, match="module does not expose a 'create_pipeline' function"):
        pipelines = find_pipelines()
    assert set(pipelines) == pipeline_names | {'__default__'}
    assert sum(pipelines.values()).outputs() == pipeline_names

@pytest.mark.parametrize('mock_package_name_with_pipelines,pipeline_names', [(x, x) for x in [set(), {'my_pipeline'}]], indirect=True)
def test_find_pipelines_skips_hidden_modules(mock_package_name_with_pipelines, pipeline_names):
    if False:
        print('Hello World!')
    pipelines_dir = Path(sys.path[0]) / mock_package_name_with_pipelines / 'pipelines'
    pipeline_dir = pipelines_dir / '.ipynb_checkpoints'
    pipeline_dir.mkdir()
    (pipeline_dir / '__init__.py').write_text(textwrap.dedent('\n            from __future__ import annotations\n\n            from kedro.pipeline import Pipeline, node, pipeline\n\n\n            def create_pipeline(**kwargs) -> Pipeline:\n                return pipeline([node(lambda: 1, None, "simple_pipeline")])\n            '))
    configure_project(mock_package_name_with_pipelines)
    pipelines = find_pipelines()
    assert set(pipelines) == pipeline_names | {'__default__'}
    assert sum(pipelines.values()).outputs() == pipeline_names

@pytest.mark.parametrize('mock_package_name_with_pipelines,pipeline_names', [(x, x) for x in [set(), {'my_pipeline'}]], indirect=True)
def test_find_pipelines_skips_modules_with_unexpected_return_value_type(mock_package_name_with_pipelines, pipeline_names):
    if False:
        print('Hello World!')
    pipelines_dir = Path(sys.path[0]) / mock_package_name_with_pipelines / 'pipelines'
    pipeline_dir = pipelines_dir / 'not_my_pipeline'
    pipeline_dir.mkdir()
    (pipeline_dir / '__init__.py').write_text(textwrap.dedent('\n            from __future__ import annotations\n\n            from kedro.pipeline import Pipeline, node, pipeline\n\n\n            def create_pipeline(**kwargs) -> dict[str, Pipeline]:\n                return {\n                    "pipe1": pipeline([node(lambda: 1, None, "pipe1")]),\n                    "pipe2": pipeline([node(lambda: 2, None, "pipe2")]),\n                }\n            '))
    configure_project(mock_package_name_with_pipelines)
    with pytest.warns(UserWarning, match="Expected the 'create_pipeline' function in the '\\S+' module to return a 'Pipeline' object, got 'dict' instead."):
        pipelines = find_pipelines()
    assert set(pipelines) == pipeline_names | {'__default__'}
    assert sum(pipelines.values()).outputs() == pipeline_names

@pytest.mark.parametrize('mock_package_name_with_pipelines,pipeline_names', [(x, x) for x in [set(), {'my_pipeline'}]], indirect=True)
def test_find_pipelines_skips_regular_files_within_the_pipelines_folder(mock_package_name_with_pipelines, pipeline_names):
    if False:
        print('Hello World!')
    pipelines_dir = Path(sys.path[0]) / mock_package_name_with_pipelines / 'pipelines'
    (pipelines_dir / 'not_my_pipeline.py').touch()
    configure_project(mock_package_name_with_pipelines)
    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=UserWarning)
        pipelines = find_pipelines()
    assert set(pipelines) == pipeline_names | {'__default__'}
    assert sum(pipelines.values()).outputs() == pipeline_names

@pytest.mark.parametrize('mock_package_name_with_pipelines,pipeline_names', [(x, x) for x in [set(), {'my_pipeline'}]], indirect=True)
def test_find_pipelines_skips_modules_that_cause_exceptions_upon_import(mock_package_name_with_pipelines, pipeline_names):
    if False:
        while True:
            i = 10
    pipelines_dir = Path(sys.path[0]) / mock_package_name_with_pipelines / 'pipelines'
    pipeline_dir = pipelines_dir / 'boulevard_of_broken_pipelines'
    pipeline_dir.mkdir()
    (pipeline_dir / '__init__.py').write_text('I walk a lonely road...')
    configure_project(mock_package_name_with_pipelines)
    with pytest.warns(UserWarning, match="An error occurred while importing the '\\S+' module."):
        pipelines = find_pipelines()
    assert set(pipelines) == pipeline_names | {'__default__'}
    assert sum(pipelines.values()).outputs() == pipeline_names

@pytest.mark.parametrize('mock_package_name_with_pipelines,pipeline_names', [(x, x) for x in [set(), {'my_pipeline'}]], indirect=True)
def test_find_pipelines_handles_simplified_project_structure(mock_package_name_with_pipelines, pipeline_names):
    if False:
        for i in range(10):
            print('nop')
    (Path(sys.path[0]) / mock_package_name_with_pipelines / 'pipeline.py').write_text(textwrap.dedent('\n            from kedro.pipeline import Pipeline, node, pipeline\n\n\n            def create_pipeline(**kwargs) -> Pipeline:\n                return pipeline([node(lambda: 1, None, "simple_pipeline")])\n            '))
    configure_project(mock_package_name_with_pipelines)
    pipelines = find_pipelines()
    assert set(pipelines) == pipeline_names | {'__default__'}
    assert sum(pipelines.values()).outputs() == pipeline_names | {'simple_pipeline'}

@pytest.mark.parametrize('mock_package_name_with_pipelines,pipeline_names', [(x, x) for x in [set(), {'my_pipeline'}]], indirect=True)
def test_find_pipelines_skips_unimportable_pipeline_module(mock_package_name_with_pipelines, pipeline_names):
    if False:
        return 10
    (Path(sys.path[0]) / mock_package_name_with_pipelines / 'pipeline.py').write_text(textwrap.dedent(f"""\n            import {''.join(pipeline_names)}\n\n            from kedro.pipeline import Pipeline, node, pipeline\n\n\n            def create_pipeline(**kwargs) -> Pipeline:\n                return pipeline([node(lambda: 1, None, "simple_pipeline")])\n            """))
    configure_project(mock_package_name_with_pipelines)
    with pytest.warns(UserWarning, match="An error occurred while importing the '\\S+' module."):
        pipelines = find_pipelines()
    assert set(pipelines) == pipeline_names | {'__default__'}
    assert sum(pipelines.values()).outputs() == pipeline_names

@pytest.mark.parametrize('mock_package_name_with_pipelines,simplified', [(set(), False), (set(), True)], indirect=['mock_package_name_with_pipelines'])
def test_find_pipelines_handles_project_structure_without_pipelines_dir(mock_package_name_with_pipelines, simplified):
    if False:
        return 10
    pipelines_dir = Path(sys.path[0]) / mock_package_name_with_pipelines / 'pipelines'
    shutil.rmtree(pipelines_dir)
    if simplified:
        (Path(sys.path[0]) / mock_package_name_with_pipelines / 'pipeline.py').write_text(textwrap.dedent('\n                from kedro.pipeline import Pipeline, node, pipeline\n\n\n                def create_pipeline(**kwargs) -> Pipeline:\n                    return pipeline([node(lambda: 1, None, "simple_pipeline")])\n                '))
    configure_project(mock_package_name_with_pipelines)
    pipelines = find_pipelines()
    assert set(pipelines) == {'__default__'}
    assert sum(pipelines.values()).outputs() == ({'simple_pipeline'} if simplified else set())