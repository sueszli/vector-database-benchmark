import pytest
from ludwig.schema.hyperopt.search_algorithm import BaseSearchAlgorithmConfig
from ludwig.schema.hyperopt.utils import register_search_algorithm_config, search_algorithm_config_registry
from ludwig.schema.utils import ludwig_dataclass, ProtectedString

@pytest.fixture(params=[('no_deps', None, False), ('installed', [('ludwig', 'ludwig')], False), ('multiple_installed', [('ludwig', 'ludwig'), ('marshmallow', 'marshmallow')], False), ('not_installed', [('fake_dependency', 'fake_dependency')], True), ('mixed_installed', [('fake_dependency', 'fake_dependency'), ('ludwig', 'ludwig')], True)])
def dependency_check_config(request):
    if False:
        for i in range(10):
            print('nop')
    (key, deps, raises_exception) = request.param

    @register_search_algorithm_config(key, dependencies=deps)
    @ludwig_dataclass
    class DependencyCheckConfig(BaseSearchAlgorithmConfig):
        type: str = ProtectedString(key)
    yield (DependencyCheckConfig(), raises_exception)
    del search_algorithm_config_registry[key]

def test_dependency_check(dependency_check_config):
    if False:
        while True:
            i = 10
    'Test that the hyperopt search alg dependency check properly identifies missing dependencies.\n\n    Most search algorithms supported by Ray Tune have additional dependencies that may not be installed. The schema\n    records these dependencies and can be used to verify they are installed at run time.\n    '
    (config, raises_exception) = dependency_check_config
    if raises_exception:
        with pytest.raises(ImportError):
            config.dependencies_installed()
    else:
        assert config.dependencies_installed()