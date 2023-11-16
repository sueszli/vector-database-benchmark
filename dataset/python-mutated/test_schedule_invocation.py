import datetime
from typing import cast
import pytest
from dagster import DagsterInstance, DagsterInvariantViolationError, DynamicPartitionsDefinition, RunRequest, StaticPartitionsDefinition, build_schedule_context, job, repository, schedule
from dagster._config.pythonic_config import ConfigurableResource
from dagster._core.errors import DagsterInvalidDefinitionError, DagsterInvalidInvocationError
from dagster._core.storage.tags import PARTITION_NAME_TAG
from dagster._core.test_utils import instance_for_test

def cron_test_schedule_factory_context():
    if False:
        for i in range(10):
            print('nop')

    @schedule(cron_schedule='* * * * *', job_name='no_pipeline')
    def basic_schedule(_):
        if False:
            return 10
        return {}
    return basic_schedule

def cron_test_schedule_factory_no_context():
    if False:
        return 10

    @schedule(cron_schedule='* * * * *', job_name='no_pipeline')
    def basic_schedule():
        if False:
            i = 10
            return i + 15
        return {}
    return basic_schedule

def test_cron_schedule_invocation_all_args():
    if False:
        while True:
            i = 10
    basic_schedule_context = cron_test_schedule_factory_context()
    assert basic_schedule_context(None) == {}
    assert basic_schedule_context(build_schedule_context()) == {}
    assert basic_schedule_context(_=None) == {}
    assert basic_schedule_context(_=build_schedule_context()) == {}
    basic_schedule_no_context = cron_test_schedule_factory_no_context()
    assert basic_schedule_no_context() == {}

def test_incorrect_cron_schedule_invocation():
    if False:
        while True:
            i = 10
    basic_schedule = cron_test_schedule_factory_context()
    with pytest.raises(DagsterInvalidInvocationError, match='Schedule evaluation function expected context argument, but no context argument was provided when invoking.'):
        basic_schedule()
    with pytest.raises(DagsterInvalidInvocationError, match="Schedule invocation expected argument '_'."):
        basic_schedule(foo=None)

def test_instance_access():
    if False:
        while True:
            i = 10
    with pytest.raises(DagsterInvariantViolationError, match='Attempted to initialize dagster instance, but no instance reference was provided.'):
        build_schedule_context().instance
    with instance_for_test() as instance:
        assert isinstance(build_schedule_context(instance).instance, DagsterInstance)

def test_schedule_invocation_resources() -> None:
    if False:
        print('Hello World!')

    class MyResource(ConfigurableResource):
        a_str: str

    @schedule(job_name='foo_job', cron_schedule='* * * * *')
    def basic_schedule_resource_req(my_resource: MyResource):
        if False:
            print('Hello World!')
        return RunRequest(run_key=None, run_config={'foo': my_resource.a_str}, tags={})
    with pytest.raises(DagsterInvalidDefinitionError, match="Resource with key 'my_resource' required by schedule 'basic_schedule_resource_req' was not provided."):
        basic_schedule_resource_req()
    with pytest.raises(DagsterInvalidDefinitionError, match="Resource with key 'my_resource' required by schedule 'basic_schedule_resource_req' was not provided."):
        basic_schedule_resource_req(build_schedule_context())
    assert hasattr(build_schedule_context(resources={'my_resource': MyResource(a_str='foo')}).resources, 'my_resource')
    assert cast(RunRequest, basic_schedule_resource_req(build_schedule_context(resources={'my_resource': MyResource(a_str='foo')}))).run_config == {'foo': 'foo'}

def test_schedule_invocation_resources_direct() -> None:
    if False:
        print('Hello World!')

    class MyResource(ConfigurableResource):
        a_str: str

    @schedule(job_name='foo_job', cron_schedule='* * * * *')
    def basic_schedule_resource_req(my_resource: MyResource):
        if False:
            print('Hello World!')
        return RunRequest(run_key=None, run_config={'foo': my_resource.a_str}, tags={})
    with pytest.raises(DagsterInvalidDefinitionError, match="Resource with key 'my_resource' required by schedule 'basic_schedule_resource_req' was not provided."):
        basic_schedule_resource_req()
    assert cast(RunRequest, basic_schedule_resource_req(context=build_schedule_context(resources={'my_resource': MyResource(a_str='foo')}))).run_config == {'foo': 'foo'}
    assert cast(RunRequest, basic_schedule_resource_req(my_resource=MyResource(a_str='foo'))).run_config == {'foo': 'foo'}
    with pytest.raises(DagsterInvalidInvocationError, match='If directly invoking a schedule, you may not provide resources as positional arguments, only as keyword arguments.'):
        assert cast(RunRequest, basic_schedule_resource_req(MyResource(a_str='foo'))).run_config == {'foo': 'foo'}
    assert cast(RunRequest, basic_schedule_resource_req(build_schedule_context(), my_resource=MyResource(a_str='foo'))).run_config == {'foo': 'foo'}

    @schedule(job_name='foo_job', cron_schedule='* * * * *')
    def basic_schedule_with_context_resource_req(my_resource: MyResource, context):
        if False:
            for i in range(10):
                print('nop')
        return RunRequest(run_key=None, run_config={'foo': my_resource.a_str}, tags={})
    assert cast(RunRequest, basic_schedule_with_context_resource_req(build_schedule_context(), my_resource=MyResource(a_str='foo'))).run_config == {'foo': 'foo'}

def test_recreating_schedule_with_resource_arg() -> None:
    if False:
        while True:
            i = 10

    class MyResource(ConfigurableResource):
        a_str: str

    @schedule(job_name='foo_job', cron_schedule='* * * * *')
    def basic_schedule_with_context_resource_req(my_resource: MyResource, context):
        if False:
            while True:
                i = 10
        return RunRequest(run_key=None, run_config={'foo': my_resource.a_str}, tags={})

    @job
    def junk_job():
        if False:
            while True:
                i = 10
        pass
    updated_schedule = basic_schedule_with_context_resource_req.with_updated_job(junk_job)
    assert cast(RunRequest, updated_schedule(build_schedule_context(), my_resource=MyResource(a_str='foo'))).run_config == {'foo': 'foo'}

def test_schedule_invocation_resources_direct_many() -> None:
    if False:
        return 10

    class MyResource(ConfigurableResource):
        a_str: str

    @schedule(job_name='foo_job', cron_schedule='* * * * *')
    def basic_schedule_resource_req(my_resource: MyResource, my_other_resource: MyResource):
        if False:
            for i in range(10):
                print('nop')
        return RunRequest(run_key=None, run_config={'foo': my_resource.a_str, 'bar': my_other_resource.a_str}, tags={})
    assert cast(RunRequest, basic_schedule_resource_req(my_other_resource=MyResource(a_str='bar'), my_resource=MyResource(a_str='foo'))).run_config == {'foo': 'foo', 'bar': 'bar'}
    assert cast(RunRequest, basic_schedule_resource_req(context=build_schedule_context(resources={'my_other_resource': MyResource(a_str='bar')}), my_resource=MyResource(a_str='foo'))).run_config == {'foo': 'foo', 'bar': 'bar'}

def test_partition_key_run_request_schedule():
    if False:
        return 10

    @job(partitions_def=StaticPartitionsDefinition(['a']))
    def my_job():
        if False:
            for i in range(10):
                print('nop')
        pass

    @schedule(cron_schedule='* * * * *', job_name='my_job')
    def my_schedule():
        if False:
            return 10
        return RunRequest(partition_key='a')

    @repository
    def my_repo():
        if False:
            for i in range(10):
                print('nop')
        return [my_job, my_schedule]
    with build_schedule_context(repository_def=my_repo, scheduled_execution_time=datetime.datetime(2023, 1, 1)) as context:
        run_requests = my_schedule.evaluate_tick(context).run_requests
        assert len(run_requests) == 1
        run_request = run_requests[0]
        assert run_request.tags.get(PARTITION_NAME_TAG) == 'a'

def test_dynamic_partition_run_request_schedule():
    if False:
        while True:
            i = 10

    @job(partitions_def=DynamicPartitionsDefinition(lambda _: ['1']))
    def my_job():
        if False:
            for i in range(10):
                print('nop')
        pass

    @schedule(cron_schedule='* * * * *', job_name='my_job')
    def my_schedule():
        if False:
            i = 10
            return i + 15
        yield RunRequest(partition_key='1', run_key='1')
        yield my_job.run_request_for_partition(partition_key='1', run_key='2')

    @repository
    def my_repo():
        if False:
            i = 10
            return i + 15
        return [my_job, my_schedule]
    with build_schedule_context(repository_def=my_repo, scheduled_execution_time=datetime.datetime(2023, 1, 1)) as context:
        run_requests = my_schedule.evaluate_tick(context).run_requests
        assert len(run_requests) == 2
        for request in run_requests:
            assert request.tags.get(PARTITION_NAME_TAG) == '1'