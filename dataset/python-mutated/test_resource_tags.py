from dagster._core.definitions.decorators import op
from dagster_dask.executor import get_dask_resource_requirements

def test_resource_tags():
    if False:
        return 10

    @op(tags={'dagster-dask/resource_requirements': {'GPU': 1, 'MEMORY': 10000000000.0}})
    def boop(_):
        if False:
            i = 10
            return i + 15
        pass
    reqs = get_dask_resource_requirements(boop.tags)
    assert reqs['GPU'] == 1
    assert reqs['MEMORY'] == 10000000000.0