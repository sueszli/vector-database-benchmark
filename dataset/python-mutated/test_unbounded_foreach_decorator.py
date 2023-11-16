import os
import subprocess
import sys
from metaflow.cli_args import cli_args
from metaflow.decorators import StepDecorator
from metaflow.exception import MetaflowException
from metaflow.unbounded_foreach import UnboundedForeachInput, UBF_CONTROL, UBF_TASK
from metaflow.util import to_unicode

class InternalTestUnboundedForeachInput(UnboundedForeachInput):
    """
    Test class that wraps around values (any iterator) and simulates an
    unbounded-foreach instead of a bounded foreach.
    """
    NAME = 'InternalTestUnboundedForeachInput'

    def __init__(self, iterable):
        if False:
            return 10
        self.iterable = iterable
        super(InternalTestUnboundedForeachInput, self).__init__()

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return iter(self.iterable)

    def __next__(self):
        if False:
            for i in range(10):
                print('nop')
        return next(self.iter)

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        if key is None:
            return self
        return self.iterable[key]

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.iterable)

    def __str__(self):
        if False:
            return 10
        return str(self.iterable)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '%s(%s)' % (self.NAME, self.iterable)

class InternalTestUnboundedForeachDecorator(StepDecorator):
    name = 'unbounded_test_foreach_internal'
    results_dict = {}

    def __init__(self, attributes=None, statically_defined=False):
        if False:
            print('Hello World!')
        super(InternalTestUnboundedForeachDecorator, self).__init__(attributes, statically_defined)

    def step_init(self, flow, graph, step_name, decorators, environment, flow_datastore, logger):
        if False:
            while True:
                i = 10
        self.environment = environment

    def control_task_step_func(self, flow, graph, retry_count):
        if False:
            for i in range(10):
                print('nop')
        from metaflow import current
        run_id = current.run_id
        step_name = current.step_name
        control_task_id = current.task_id
        (_, split_step_name, split_task_id) = control_task_id.split('-')[1:]
        env_to_use = getattr(self.environment, 'base_env', self.environment)
        executable = env_to_use.executable(step_name)
        script = sys.argv[0]
        assert flow._unbounded_foreach
        foreach_iter = flow.input
        if not isinstance(foreach_iter, InternalTestUnboundedForeachInput):
            raise MetaflowException('Expected type to be InternalTestUnboundedForeachInput. Found %s' % type(foreach_iter))
        foreach_num_splits = sum((1 for _ in foreach_iter))
        print('Simulating UnboundedForeach over value:', foreach_iter, 'num_splits:', foreach_num_splits)
        mapper_tasks = []
        for i in range(foreach_num_splits):
            task_id = '%s-%d' % (control_task_id.replace('control-', 'test-ubf-'), i)
            pathspec = '%s/%s/%s' % (run_id, step_name, task_id)
            mapper_tasks.append(to_unicode(pathspec))
            input_paths = '%s/%s/%s' % (run_id, split_step_name, split_task_id)
            kwargs = cli_args.step_kwargs
            kwargs['split_index'] = str(i)
            kwargs['run_id'] = run_id
            kwargs['task_id'] = task_id
            kwargs['input_paths'] = input_paths
            kwargs['ubf_context'] = UBF_TASK
            kwargs['retry_count'] = 0
            cmd = cli_args.step_command(executable, script, step_name, step_kwargs=kwargs)
            step_cli = ' '.join(cmd)
            print('[${cwd}] Starting split#{split} with cmd:{cmd}'.format(cwd=os.getcwd(), split=i, cmd=step_cli))
            output_bytes = subprocess.check_output(cmd)
            output = to_unicode(output_bytes)
            for line in output.splitlines():
                print('[Split#%d] %s' % (i, line))
        flow._control_mapper_tasks = mapper_tasks

    def task_decorate(self, step_func, flow, graph, retry_count, max_user_code_retries, ubf_context):
        if False:
            print('Hello World!')
        if ubf_context == UBF_CONTROL:
            from functools import partial
            return partial(self.control_task_step_func, flow, graph, retry_count)
        else:
            return step_func

    def step_task_retry_count(self):
        if False:
            while True:
                i = 10
        return (None, None)