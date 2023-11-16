import importlib
import json
import os
import platform
import sys
import tempfile
from metaflow.decorators import FlowDecorator, StepDecorator
from metaflow.extension_support import EXT_PKG
from metaflow.metaflow_environment import InvalidEnvironmentException
from metaflow.util import get_metaflow_root
from ... import INFO_FILE

class CondaStepDecorator(StepDecorator):
    """
    Specifies the Conda environment for the step.

    Information in this decorator will augment any
    attributes set in the `@conda_base` flow-level decorator. Hence,
    you can use `@conda_base` to set packages required by all
    steps and use `@conda` to specify step-specific overrides.

    Parameters
    ----------
    packages : Dict[str, str], default: {}
        Packages to use for this step. The key is the name of the package
        and the value is the version to use.
    libraries : Dict[str, str], default: {}
        Supported for backward compatibility. When used with packages, packages will take precedence.
    python : str, optional
        Version of Python to use, e.g. '3.7.4'. A default value of None implies
        that the version used will correspond to the version of the Python interpreter used to start the run.
    disabled : bool, default: False
        If set to True, disables @conda.
    """
    name = 'conda'
    defaults = {'packages': {}, 'libraries': {}, 'python': None, 'disabled': None}

    def __init__(self, attributes=None, statically_defined=False):
        if False:
            while True:
                i = 10
        super(CondaStepDecorator, self).__init__(attributes, statically_defined)
        self.attributes['packages'] = {**self.attributes['libraries'], **self.attributes['packages']}
        del self.attributes['libraries']

    def step_init(self, flow, graph, step, decos, environment, flow_datastore, logger):
        if False:
            print('Hello World!')
        self.flow = flow
        self.step = step
        self.environment = environment
        self.datastore = flow_datastore
        if 'conda_base' in self.flow._flow_decorators:
            super_attributes = self.flow._flow_decorators['conda_base'][0].attributes
            self.attributes['packages'] = {**super_attributes['packages'], **self.attributes['packages']}
            self.attributes['python'] = self.attributes['python'] or super_attributes['python']
            self.attributes['disabled'] = self.attributes['disabled'] if self.attributes['disabled'] is not None else super_attributes['disabled']
        if not self.attributes['disabled']:
            self.attributes['disabled'] = False
        if not self.attributes['python']:
            self.attributes['python'] = platform.python_version()
        _supported_virtual_envs = ['conda']
        _supported_virtual_envs.extend(['pypi'])
        if environment.TYPE not in _supported_virtual_envs:
            raise InvalidEnvironmentException('@%s decorator requires %s' % (self.name, ' or '.join(['--environment=%s' % env for env in _supported_virtual_envs])))
        from metaflow.plugins.datastores.local_storage import LocalStorage
        environment.set_local_root(LocalStorage.get_datastore_root_from_config(logger))
        self.disabled = self.environment.is_disabled(next((step for step in self.flow if step.name == self.step)))

    def runtime_init(self, flow, graph, package, run_id):
        if False:
            for i in range(10):
                print('nop')
        if self.disabled:
            return
        self.metaflow_dir = tempfile.TemporaryDirectory(dir='/tmp')
        os.symlink(os.path.join(get_metaflow_root(), 'metaflow'), os.path.join(self.metaflow_dir.name, 'metaflow'))
        info = os.path.join(get_metaflow_root(), os.path.basename(INFO_FILE))
        if os.path.isfile(info):
            os.symlink(info, os.path.join(self.metaflow_dir.name, os.path.basename(INFO_FILE)))
        else:
            with open(os.path.join(self.metaflow_dir.name, os.path.basename(INFO_FILE)), mode='wt', encoding='utf-8') as f:
                f.write(json.dumps(self.environment.get_environment_info(include_ext_info=True)))
        self.addl_paths = None
        try:
            m = importlib.import_module(EXT_PKG)
        except ImportError:
            pass
        else:
            custom_paths = list(set(m.__path__))
            if len(custom_paths) == 1:
                os.symlink(custom_paths[0], os.path.join(self.metaflow_dir.name, EXT_PKG))
            else:
                self.addl_paths = []
                for p in custom_paths:
                    temp_dir = tempfile.mkdtemp(dir=self.metaflow_dir.name)
                    os.symlink(p, os.path.join(temp_dir, EXT_PKG))
                    self.addl_paths.append(temp_dir)

    def runtime_task_created(self, task_datastore, task_id, split_index, input_paths, is_cloned, ubf_context):
        if False:
            for i in range(10):
                print('nop')
        if self.disabled:
            return
        self.interpreter = self.environment.interpreter(self.step) if not any((decorator.name in ['batch', 'kubernetes'] for decorator in next((step for step in self.flow if step.name == self.step)).decorators)) else None

    def task_pre_step(self, step_name, task_datastore, meta, run_id, task_id, flow, graph, retry_count, max_retries, ubf_context, inputs):
        if False:
            while True:
                i = 10
        if self.disabled:
            return
        os.environ['PATH'] = os.pathsep.join(filter(None, (os.path.dirname(os.path.realpath(sys.executable)), os.environ.get('PATH'))))

    def runtime_step_cli(self, cli_args, retry_count, max_user_code_retries, ubf_context):
        if False:
            while True:
                i = 10
        if self.disabled:
            return
        python_path = self.metaflow_dir.name
        if self.addl_paths is not None:
            addl_paths = os.pathsep.join(self.addl_paths)
            python_path = os.pathsep.join([addl_paths, python_path])
        cli_args.env['PYTHONPATH'] = python_path
        if self.interpreter:
            cli_args.env['PYTHONNOUSERSITE'] = '1'
            cli_args.entrypoint[0] = self.interpreter

    def runtime_finished(self, exception):
        if False:
            for i in range(10):
                print('nop')
        if self.disabled:
            return
        self.metaflow_dir.cleanup()

class CondaFlowDecorator(FlowDecorator):
    """
    Specifies the Conda environment for all steps of the flow.

    Use `@conda_base` to set common libraries required by all
    steps and use `@conda` to specify step-specific additions.

    Parameters
    ----------
    packages : Dict[str, str], default: {}
        Packages to use for this flow. The key is the name of the package
        and the value is the version to use.
    libraries : Dict[str, str], default: {}
        Supported for backward compatibility. When used with packages, packages will take precedence.
    python : str, optional
        Version of Python to use, e.g. '3.7.4'. A default value of None implies
        that the version used will correspond to the version of the Python interpreter used to start the run.
    disabled : bool, default: False
        If set to True, disables Conda.
    """
    name = 'conda_base'
    defaults = {'packages': {}, 'libraries': {}, 'python': None, 'disabled': None}

    def __init__(self, attributes=None, statically_defined=False):
        if False:
            return 10
        super(CondaFlowDecorator, self).__init__(attributes, statically_defined)
        self.attributes['packages'] = {**self.attributes['libraries'], **self.attributes['packages']}
        del self.attributes['libraries']
        if self.attributes['python']:
            self.attributes['python'] = str(self.attributes['python'])

    def flow_init(self, flow, graph, environment, flow_datastore, metadata, logger, echo, options):
        if False:
            while True:
                i = 10
        _supported_virtual_envs = ['conda']
        _supported_virtual_envs.extend(['pypi'])
        if environment.TYPE not in _supported_virtual_envs:
            raise InvalidEnvironmentException('@%s decorator requires %s' % (self.name, ' or '.join(['--environment=%s' % env for env in _supported_virtual_envs])))