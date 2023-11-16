import contextlib
import inspect
import os
import sys
import threading
import time
import uuid
from collections.abc import Sized
from functools import wraps
from timeit import default_timer as timer
from typing import Any, Callable, List, Optional, Set, TypeVar, Union, cast, overload
from typing_extensions import Final
from streamlit import config, util
from streamlit.logger import get_logger
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.proto.PageProfile_pb2 import Argument, Command
_LOGGER = get_logger(__name__)
_MAX_TRACKED_COMMANDS: Final = 200
_MAX_TRACKED_PER_COMMAND: Final = 25
_OBJECT_NAME_MAPPING: Final = {'streamlit.delta_generator.DeltaGenerator': 'DG', 'pandas.core.frame.DataFrame': 'DataFrame', 'plotly.graph_objs._figure.Figure': 'PlotlyFigure', 'bokeh.plotting.figure.Figure': 'BokehFigure', 'matplotlib.figure.Figure': 'MatplotlibFigure', 'pandas.io.formats.style.Styler': 'PandasStyler', 'pandas.core.indexes.base.Index': 'PandasIndex', 'pandas.core.series.Series': 'PandasSeries', 'streamlit.connections.snowpark_connection.SnowparkConnection': 'SnowparkConnection', 'streamlit.connections.sql_connection.SQLConnection': 'SQLConnection'}
_ATTRIBUTIONS_TO_CHECK: Final = ['pymysql', 'MySQLdb', 'mysql', 'pymongo', 'ibis', 'boto3', 'psycopg2', 'psycopg3', 'sqlalchemy', 'elasticsearch', 'pyodbc', 'pymssql', 'cassandra', 'azure', 'redis', 'sqlite3', 'neo4j', 'duckdb', 'opensearchpy', 'supabase', 'polars', 'dask', 'vaex', 'modin', 'pyspark', 'cudf', 'xarray', 'ray', 'openai', 'langchain', 'llama_index', 'llama_cpp', 'anthropic', 'pyllamacpp', 'cohere', 'transformers', 'nomic', 'diffusers', 'semantic_kernel', 'replicate', 'huggingface_hub', 'wandb', 'torch', 'tensorflow', 'trubrics', 'comet_ml', 'clarifai', 'reka', 'hegel', 'fastchat', 'assemblyai', 'openllm', 'embedchain', 'haystack', 'vllm', 'alpa', 'jinaai', 'guidance', 'litellm', 'comet_llm', 'instructor', 'prefect', 'luigi', 'airflow', 'dagster', 'pgvector', 'faiss', 'annoy', 'pinecone', 'chromadb', 'weaviate', 'qdrant_client', 'pymilvus', 'lancedb', 'datasets', 'snowflake', 'streamlit_extras', 'streamlit_pydantic', 'pydantic', 'plost']
_ETC_MACHINE_ID_PATH = '/etc/machine-id'
_DBUS_MACHINE_ID_PATH = '/var/lib/dbus/machine-id'

def _get_machine_id_v3() -> str:
    if False:
        while True:
            i = 10
    "Get the machine ID\n\n    This is a unique identifier for a user for tracking metrics in Segment,\n    that is broken in different ways in some Linux distros and Docker images.\n    - at times just a hash of '', which means many machines map to the same ID\n    - at times a hash of the same string, when running in a Docker container\n    "
    machine_id = str(uuid.getnode())
    if os.path.isfile(_ETC_MACHINE_ID_PATH):
        with open(_ETC_MACHINE_ID_PATH, 'r') as f:
            machine_id = f.read()
    elif os.path.isfile(_DBUS_MACHINE_ID_PATH):
        with open(_DBUS_MACHINE_ID_PATH, 'r') as f:
            machine_id = f.read()
    return machine_id

class Installation:
    _instance_lock = threading.Lock()
    _instance: Optional['Installation'] = None

    @classmethod
    def instance(cls) -> 'Installation':
        if False:
            return 10
        'Returns the singleton Installation'
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = Installation()
        return cls._instance

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.installation_id_v3 = str(uuid.uuid5(uuid.NAMESPACE_DNS, _get_machine_id_v3()))

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return util.repr_(self)

    @property
    def installation_id(self):
        if False:
            print('Hello World!')
        return self.installation_id_v3

def _get_type_name(obj: object) -> str:
    if False:
        i = 10
        return i + 15
    'Get a simplified name for the type of the given object.'
    with contextlib.suppress(Exception):
        obj_type = obj if inspect.isclass(obj) else type(obj)
        type_name = 'unknown'
        if hasattr(obj_type, '__qualname__'):
            type_name = obj_type.__qualname__
        elif hasattr(obj_type, '__name__'):
            type_name = obj_type.__name__
        if obj_type.__module__ != 'builtins':
            type_name = f'{obj_type.__module__}.{type_name}'
        if type_name in _OBJECT_NAME_MAPPING:
            type_name = _OBJECT_NAME_MAPPING[type_name]
        return type_name
    return 'failed'

def _get_top_level_module(func: Callable[..., Any]) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Get the top level module for the given function.'
    module = inspect.getmodule(func)
    if module is None or not module.__name__:
        return 'unknown'
    return module.__name__.split('.')[0]

def _get_arg_metadata(arg: object) -> Optional[str]:
    if False:
        while True:
            i = 10
    'Get metadata information related to the value of the given object.'
    with contextlib.suppress(Exception):
        if isinstance(arg, bool):
            return f'val:{arg}'
        if isinstance(arg, Sized):
            return f'len:{len(arg)}'
    return None

def _get_command_telemetry(_command_func: Callable[..., Any], _command_name: str, *args, **kwargs) -> Command:
    if False:
        return 10
    'Get telemetry information for the given callable and its arguments.'
    arg_keywords = inspect.getfullargspec(_command_func).args
    self_arg: Optional[Any] = None
    arguments: List[Argument] = []
    is_method = inspect.ismethod(_command_func)
    name = _command_name
    for (i, arg) in enumerate(args):
        pos = i
        if is_method:
            i = i + 1
        keyword = arg_keywords[i] if len(arg_keywords) > i else f'{i}'
        if keyword == 'self':
            self_arg = arg
            continue
        argument = Argument(k=keyword, t=_get_type_name(arg), p=pos)
        arg_metadata = _get_arg_metadata(arg)
        if arg_metadata:
            argument.m = arg_metadata
        arguments.append(argument)
    for (kwarg, kwarg_value) in kwargs.items():
        argument = Argument(k=kwarg, t=_get_type_name(kwarg_value))
        arg_metadata = _get_arg_metadata(kwarg_value)
        if arg_metadata:
            argument.m = arg_metadata
        arguments.append(argument)
    top_level_module = _get_top_level_module(_command_func)
    if top_level_module != 'streamlit':
        name = f'external:{top_level_module}:{name}'
    if name == 'create_instance' and self_arg and hasattr(self_arg, 'name') and self_arg.name:
        name = f'component:{self_arg.name}'
    return Command(name=name, args=arguments)

def to_microseconds(seconds: float) -> int:
    if False:
        i = 10
        return i + 15
    'Convert seconds into microseconds.'
    return int(seconds * 1000000)
F = TypeVar('F', bound=Callable[..., Any])

@overload
def gather_metrics(name: str, func: F) -> F:
    if False:
        while True:
            i = 10
    ...

@overload
def gather_metrics(name: str, func: None=None) -> Callable[[F], F]:
    if False:
        while True:
            i = 10
    ...

def gather_metrics(name: str, func: Optional[F]=None) -> Union[Callable[[F], F], F]:
    if False:
        i = 10
        return i + 15
    'Function decorator to add telemetry tracking to commands.\n\n    Parameters\n    ----------\n    func : callable\n    The function to track for telemetry.\n\n    name : str or None\n    Overwrite the function name with a custom name that is used for telemetry tracking.\n\n    Example\n    -------\n    >>> @st.gather_metrics\n    ... def my_command(url):\n    ...     return url\n\n    >>> @st.gather_metrics(name="custom_name")\n    ... def my_command(url):\n    ...     return url\n    '
    if not name:
        _LOGGER.warning('gather_metrics: name is empty')
        name = 'undefined'
    if func is None:

        def wrapper(f: F) -> F:
            if False:
                i = 10
                return i + 15
            return gather_metrics(name=name, func=f)
        return wrapper
    else:
        non_optional_func = func

    @wraps(non_optional_func)
    def wrapped_func(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        exec_start = timer()
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        from streamlit.runtime.scriptrunner.script_runner import RerunException
        ctx = get_script_run_ctx(suppress_warning=True)
        tracking_activated = ctx is not None and ctx.gather_usage_stats and (not ctx.command_tracking_deactivated) and (len(ctx.tracked_commands) < _MAX_TRACKED_COMMANDS)
        deferred_exception: Optional[RerunException] = None
        command_telemetry: Optional[Command] = None
        if ctx and tracking_activated:
            try:
                command_telemetry = _get_command_telemetry(non_optional_func, name, *args, **kwargs)
                if command_telemetry.name not in ctx.tracked_commands_counter or ctx.tracked_commands_counter[command_telemetry.name] < _MAX_TRACKED_PER_COMMAND:
                    ctx.tracked_commands.append(command_telemetry)
                ctx.tracked_commands_counter.update([command_telemetry.name])
                ctx.command_tracking_deactivated = True
            except Exception as ex:
                _LOGGER.debug('Failed to collect command telemetry', exc_info=ex)
        try:
            result = non_optional_func(*args, **kwargs)
        except RerunException as ex:
            if tracking_activated and command_telemetry:
                command_telemetry.time = to_microseconds(timer() - exec_start)
            raise ex
        finally:
            if ctx:
                ctx.command_tracking_deactivated = False
        if tracking_activated and command_telemetry:
            command_telemetry.time = to_microseconds(timer() - exec_start)
        return result
    with contextlib.suppress(AttributeError):
        wrapped_func.__dict__.update(non_optional_func.__dict__)
        wrapped_func.__signature__ = inspect.signature(non_optional_func)
    return cast(F, wrapped_func)

def create_page_profile_message(commands: List[Command], exec_time: int, prep_time: int, uncaught_exception: Optional[str]=None) -> ForwardMsg:
    if False:
        print('Hello World!')
    'Create and return the full PageProfile ForwardMsg.'
    msg = ForwardMsg()
    msg.page_profile.commands.extend(commands)
    msg.page_profile.exec_time = exec_time
    msg.page_profile.prep_time = prep_time
    msg.page_profile.headless = config.get_option('server.headless')
    config_options: Set[str] = set()
    if config._config_options:
        for option_name in config._config_options.keys():
            if not config.is_manually_set(option_name):
                continue
            config_option = config._config_options[option_name]
            if config_option.is_default:
                option_name = f'{option_name}:default'
            config_options.add(option_name)
    msg.page_profile.config.extend(config_options)
    attributions: Set[str] = {attribution for attribution in _ATTRIBUTIONS_TO_CHECK if attribution in sys.modules}
    msg.page_profile.os = str(sys.platform)
    msg.page_profile.timezone = str(time.tzname)
    msg.page_profile.attributions.extend(attributions)
    if uncaught_exception:
        msg.page_profile.uncaught_exception = uncaught_exception
    return msg