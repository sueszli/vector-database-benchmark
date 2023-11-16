import os
import pickle
import re
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, Tuple
from dagster import DynamicOut, DynamicOutput, ExecutorDefinition, JobDefinition, ReexecutionOptions, execute_job, job, op, reconstructable, resource
from dagster._core.test_utils import instance_for_test

def get_dynamic_job_resource_init_failure(executor_def: ExecutorDefinition) -> Tuple[JobDefinition, Callable[[str, int, int], Dict[str, Any]]]:
    if False:
        while True:
            i = 10

    @op(out=DynamicOut(), config_schema={'num_dynamic_steps': int})
    def source(context):
        if False:
            i = 10
            return i + 15
        for i in range(context.op_config['num_dynamic_steps']):
            yield DynamicOutput(i, mapping_key=str(i))

    @resource(config_schema={'path': str, 'allowed_initializations': int})
    def resource_for_dynamic_step(init_context):
        if False:
            for i in range(10):
                print('nop')
        with open(os.path.join(init_context.resource_config['path'], 'count.pkl'), 'rb') as f:
            init_count = pickle.load(f)
            if init_count == init_context.resource_config['allowed_initializations']:
                raise Exception('too many initializations.')
        with open(os.path.join(init_context.resource_config['path'], 'count.pkl'), 'wb') as f:
            init_count += 1
            pickle.dump(init_count, f)
        return None

    @op(required_resource_keys={'foo'})
    def mapped_op(x):
        if False:
            i = 10
            return i + 15
        pass

    @op
    def consumer(x):
        if False:
            return 10
        pass

    @job(resource_defs={'foo': resource_for_dynamic_step}, executor_def=executor_def)
    def the_job():
        if False:
            while True:
                i = 10
        consumer(source().map(mapped_op).collect())
    return (the_job, lambda temp_dir, init_count, dynamic_steps: {'resources': {'foo': {'config': {'path': temp_dir, 'allowed_initializations': init_count}}}, 'ops': {'source': {'config': {'num_dynamic_steps': dynamic_steps}}}})

def get_dynamic_job_op_failure(executor_def: ExecutorDefinition) -> Tuple[JobDefinition, Callable[[str, int, int], Dict[str, Any]]]:
    if False:
        i = 10
        return i + 15

    @op(out=DynamicOut())
    def source():
        if False:
            print('Hello World!')
        for i in range(3):
            yield DynamicOutput(i, mapping_key=str(i))

    @op(config_schema={'path': str, 'allowed_runs': int})
    def mapped_op(context, x):
        if False:
            while True:
                i = 10
        with open(os.path.join(context.op_config['path'], 'count.pkl'), 'rb') as f:
            run_count = pickle.load(f)
            if run_count == context.op_config['allowed_runs']:
                raise Exception('oof')
        with open(os.path.join(context.op_config['path'], 'count.pkl'), 'wb') as f:
            run_count += 1
            pickle.dump(run_count, f)

    @op
    def consumer(x):
        if False:
            print('Hello World!')
        return 4

    @job
    def the_job():
        if False:
            return 10
        consumer(source().map(mapped_op).collect())
    return (the_job, lambda temp_dir, run_count, dynamic_steps: {'ops': {'mapped_op': {'config': {'path': temp_dir, 'allowed_runs': run_count}}, 'source': {'config': {'num_dynamic_steps': dynamic_steps}}}})

def _regexes_match(regexes, the_list):
    if False:
        for i in range(10):
            print('nop')
    return all([re.match(regex, item) for (regex, item) in zip(regexes, the_list)])

def _write_blank_count(path):
    if False:
        for i in range(10):
            print('nop')
    with open(os.path.join(path, 'count.pkl'), 'wb') as f:
        pickle.dump(0, f)

def assert_expected_failure_behavior(job_fn, config_fn):
    if False:
        return 10
    num_dynamic_steps = 3
    with TemporaryDirectory() as temp_dir:
        with instance_for_test(temp_dir=temp_dir) as instance:
            _write_blank_count(temp_dir)
            result = execute_job(reconstructable(job_fn), instance, run_config=config_fn(temp_dir, 1, num_dynamic_steps))
            assert not result.success
            assert len(result.get_step_success_events()) == 2
            assert _regexes_match(['source', 'mapped_op\\[\\d\\]'], [event.step_key for event in result.get_step_success_events()])
            assert len(result.get_failed_step_keys()) == 2
            assert _regexes_match(['mapped_op\\[\\d\\]', 'mapped_op\\[\\d\\]'], list(result.get_failed_step_keys()))
            _write_blank_count(temp_dir)
            retry_result = execute_job(reconstructable(job_fn), instance, run_config=config_fn(temp_dir, num_dynamic_steps, num_dynamic_steps), reexecution_options=ReexecutionOptions.from_failure(run_id=result.run_id, instance=instance))
            assert retry_result.success
            assert len(retry_result.get_step_success_events()) == 3
            assert _regexes_match(['mapped_op\\[\\d\\]', 'mapped_op\\[\\d\\]', 'consumer'], [event.step_key for event in retry_result.get_step_success_events()])