import tempfile
import pytest
from dagster import ConfigurableResource, build_schedule_context, schedule
from dagster._core.instance.ref import InstanceRef

def test_schedule_instance_does_init_with_resource() -> None:
    if False:
        return 10

    class MyResource(ConfigurableResource):
        foo: str

    @schedule(cron_schedule='@daily', job_name='some_job')
    def a_schedule(context, my_resource: MyResource):
        if False:
            i = 10
            return i + 15
        raise Exception('should not execute')
    with tempfile.TemporaryDirectory() as tempdir:
        unloadable_instance_ref = InstanceRef.from_dir(tempdir, overrides={'run_storage': {'module': 'dagster._core.test_utils', 'class': 'ExplodeOnInitRunStorage', 'config': {'base_dir': 'UNUSED'}}})
        with pytest.raises(NotImplementedError, match='from_config_value was called'):
            a_schedule(build_schedule_context(instance_ref=unloadable_instance_ref, resources={'my_resource': MyResource(foo='bar')}))

def test_schedule_instance_does_no_init_with_no_resources() -> None:
    if False:
        while True:
            i = 10
    executed = {}

    @schedule(cron_schedule='@daily', job_name='some_job')
    def a_schedule(context):
        if False:
            i = 10
            return i + 15
        executed['yes'] = True
    with tempfile.TemporaryDirectory() as tempdir:
        unloadable_instance_ref = InstanceRef.from_dir(tempdir, overrides={'run_storage': {'module': 'dagster._core.test_utils', 'class': 'ExplodeOnInitRunStorage', 'config': {'base_dir': 'UNUSED'}}})
        a_schedule(build_schedule_context(instance_ref=unloadable_instance_ref))
    assert executed['yes']