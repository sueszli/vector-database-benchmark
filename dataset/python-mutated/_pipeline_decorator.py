import inspect
import logging
from collections import OrderedDict
from functools import wraps
from inspect import Parameter, signature
from pathlib import Path
from typing import Callable, Dict, List, Optional, TypeVar, Union, overload
from typing_extensions import ParamSpec
from azure.ai.ml._utils.utils import is_private_preview_enabled
from azure.ai.ml.entities import Data, Model, PipelineJob, PipelineJobSettings
from azure.ai.ml.entities._builders.pipeline import Pipeline
from azure.ai.ml.entities._inputs_outputs import Input, is_group
from azure.ai.ml.entities._job.pipeline._io import NodeOutput, PipelineInput, _GroupAttrDict
from azure.ai.ml.entities._job.pipeline._pipeline_expression import PipelineExpression
from azure.ai.ml.exceptions import MultipleValueError, ParamValueNotExistsError, TooManyPositionalArgsError, UnexpectedKeywordError, UnsupportedParameterKindError, UserErrorException
from azure.ai.ml.entities._builders import BaseNode
from azure.ai.ml.dsl._pipeline_component_builder import PipelineComponentBuilder, _is_inside_dsl_pipeline_func
from azure.ai.ml.dsl._pipeline_decorator import _validate_args
from azure.ai.ml.dsl._settings import _dsl_settings_stack
from azure.ai.ml.dsl._utils import _resolve_source_file
SUPPORTED_INPUT_TYPES = (PipelineInput, NodeOutput, Input, Model, Data, Pipeline, str, bool, int, float, PipelineExpression, _GroupAttrDict)
module_logger = logging.getLogger(__name__)
T = TypeVar('T')
P = ParamSpec('P')

@overload
def pipeline(func: None=None, *, name: Optional[str]=None, version: Optional[str]=None, display_name: Optional[str]=None, description: Optional[str]=None, experiment_name: Optional[str]=None, tags: Optional[Dict[str, str]]=None, **kwargs) -> Callable[[Callable[P, T]], Callable[P, PipelineJob]]:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def pipeline(func: Callable[P, T]=None, *, name: Optional[str]=None, version: Optional[str]=None, display_name: Optional[str]=None, description: Optional[str]=None, experiment_name: Optional[str]=None, tags: Optional[Dict[str, str]]=None, **kwargs) -> Callable[P, PipelineJob]:
    if False:
        i = 10
        return i + 15
    ...

def pipeline(func: Optional[Callable[P, T]]=None, *, name: Optional[str]=None, version: Optional[str]=None, display_name: Optional[str]=None, description: Optional[str]=None, experiment_name: Optional[str]=None, tags: Optional[Dict[str, str]]=None, **kwargs) -> Union[Callable[[Callable[P, T]], Callable[P, PipelineJob]], Callable[P, PipelineJob]]:
    if False:
        while True:
            i = 10
    'Build a pipeline which contains all component nodes defined in this function.\n\n    :param func: The user pipeline function to be decorated.\n    :type func: types.FunctionType\n    :keyword name: The name of pipeline component, defaults to function name.\n    :paramtype name: str\n    :keyword version: The version of pipeline component, defaults to "1".\n    :paramtype version: str\n    :keyword display_name: The display name of pipeline component, defaults to function name.\n    :paramtype display_name: str\n    :keyword description: The description of the built pipeline.\n    :paramtype description: str\n    :keyword experiment_name: Name of the experiment the job will be created under,                 if None is provided, experiment will be set to current directory.\n    :paramtype experiment_name: str\n    :keyword tags: The tags of pipeline component.\n    :paramtype tags: dict[str, str]\n    :keyword kwargs: A dictionary of additional configuration parameters.\n    :paramtype kwargs: dict\n\n    .. admonition:: Example:\n\n        .. literalinclude:: ../../../../samples/ml_samples_pipeline_job_configurations.py\n            :start-after: [START configure_pipeline]\n            :end-before: [END configure_pipeline]\n            :language: python\n            :dedent: 8\n            :caption: Shows how to create a pipeline using this decorator.\n    :return: Either\n      * A decorator, if `func` is None\n      * The decorated `func`\n    :rtype: Union[\n        Callable[[Callable], Callable[..., PipelineJob]],\n        Callable[P, PipelineJob]\n      ]\n    '
    get_component = kwargs.get('get_component', False)

    def pipeline_decorator(func: Callable[P, T]) -> Callable[P, PipelineJob]:
        if False:
            return 10
        if not isinstance(func, Callable):
            raise UserErrorException(f'Dsl pipeline decorator accept only function type, got {type(func)}.')
        non_pipeline_inputs = kwargs.get('non_pipeline_inputs', []) or kwargs.get('non_pipeline_parameters', [])
        compute = kwargs.get('compute', None)
        default_compute_target = kwargs.get('default_compute_target', None)
        default_compute_target = kwargs.get('default_compute', None) or default_compute_target
        continue_on_step_failure = kwargs.get('continue_on_step_failure', None)
        on_init = kwargs.get('on_init', None)
        on_finalize = kwargs.get('on_finalize', None)
        default_datastore = kwargs.get('default_datastore', None)
        force_rerun = kwargs.get('force_rerun', None)
        job_settings = {'default_datastore': default_datastore, 'continue_on_step_failure': continue_on_step_failure, 'force_rerun': force_rerun, 'default_compute': default_compute_target, 'on_init': on_init, 'on_finalize': on_finalize}
        func_entry_path = _resolve_source_file()
        if not func_entry_path:
            func_path = Path(inspect.getfile(func))
            if func_path.exists():
                func_entry_path = func_path.resolve().absolute()
        job_settings = {k: v for (k, v) in job_settings.items() if v is not None}
        pipeline_builder = PipelineComponentBuilder(func=func, name=name, version=version, display_name=display_name, description=description, default_datastore=default_datastore, tags=tags, source_path=str(func_entry_path), non_pipeline_inputs=non_pipeline_inputs)

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> PipelineJob:
            if False:
                print('Hello World!')
            _dsl_settings_stack.push()
            try:
                provided_positional_kwargs = _validate_args(func, args, kwargs, non_pipeline_inputs)
                pipeline_parameters = {k: v for (k, v) in provided_positional_kwargs.items() if k not in non_pipeline_inputs}
                pipeline_builder._update_inputs(pipeline_parameters)
                non_pipeline_params_dict = {k: v for (k, v) in provided_positional_kwargs.items() if k in non_pipeline_inputs}
                pipeline_component = pipeline_builder.build(user_provided_kwargs=provided_positional_kwargs, non_pipeline_inputs_dict=non_pipeline_params_dict, non_pipeline_inputs=non_pipeline_inputs)
            finally:
                dsl_settings = _dsl_settings_stack.pop()
            if dsl_settings.init_job_set:
                job_settings['on_init'] = dsl_settings.init_job_name(pipeline_component.jobs)
            if dsl_settings.finalize_job_set:
                job_settings['on_finalize'] = dsl_settings.finalize_job_name(pipeline_component.jobs)
            common_init_args = {'experiment_name': experiment_name, 'component': pipeline_component, 'inputs': pipeline_parameters, 'tags': tags}
            if _is_inside_dsl_pipeline_func() or get_component:
                if job_settings.get('on_init') is not None or job_settings.get('on_finalize') is not None:
                    raise UserErrorException('On_init/on_finalize is not supported for pipeline component.')
                built_pipeline = Pipeline(_from_component_func=True, **common_init_args)
                if job_settings:
                    module_logger.warning('Job settings %s on pipeline function %r are ignored when using inside PipelineJob.', job_settings, func.__name__)
            else:
                built_pipeline = PipelineJob(jobs=pipeline_component.jobs, compute=compute, settings=PipelineJobSettings(**job_settings), **common_init_args)
            return built_pipeline
        wrapper._is_dsl_func = True
        wrapper._job_settings = job_settings
        wrapper._pipeline_builder = pipeline_builder
        return wrapper
    if func is not None:
        return pipeline_decorator(func)
    return pipeline_decorator