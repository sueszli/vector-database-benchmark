import functools
import itertools
import logging
import os
import re
from dataclasses import dataclass, field
from importlib import __import__
from typing import Dict, List, Optional, Set, Union
from weakref import WeakSet
log = logging.getLogger(__name__)
DEFAULT_LOG_LEVEL = logging.WARNING
LOG_ENV_VAR = 'TORCH_LOGS'
LOG_FORMAT_ENV_VAR = 'TORCH_LOGS_FORMAT'

@dataclass
class LogRegistry:
    log_alias_to_log_qnames: Dict[str, List[str]] = field(default_factory=dict)
    artifact_log_qnames: Set[str] = field(default_factory=set)
    child_log_qnames: Set[str] = field(default_factory=set)
    artifact_names: Set[str] = field(default_factory=set)
    visible_artifacts: Set[str] = field(default_factory=set)
    artifact_descriptions: Dict[str, str] = field(default_factory=dict)
    off_by_default_artifact_names: Set[str] = field(default_factory=set)
    artifact_log_formatters: Dict[str, logging.Formatter] = field(default_factory=dict)

    def is_artifact(self, name):
        if False:
            i = 10
            return i + 15
        return name in self.artifact_names

    def is_log(self, alias):
        if False:
            while True:
                i = 10
        return alias in self.log_alias_to_log_qnames

    def register_log(self, alias, log_qnames: Union[str, List[str]]):
        if False:
            while True:
                i = 10
        if isinstance(log_qnames, str):
            log_qnames = [log_qnames]
        self.log_alias_to_log_qnames[alias] = log_qnames

    def register_artifact_name(self, name, description, visible, off_by_default, log_format):
        if False:
            i = 10
            return i + 15
        self.artifact_names.add(name)
        if visible:
            self.visible_artifacts.add(name)
        self.artifact_descriptions[name] = description
        if off_by_default:
            self.off_by_default_artifact_names.add(name)
        if log_format is not None:
            self.artifact_log_formatters[name] = logging.Formatter(log_format)

    def register_artifact_log(self, artifact_log_qname):
        if False:
            return 10
        self.artifact_log_qnames.add(artifact_log_qname)

    def register_child_log(self, log_qname):
        if False:
            print('Hello World!')
        self.child_log_qnames.add(log_qname)

    def get_log_qnames(self) -> Set[str]:
        if False:
            return 10
        return {qname for qnames in self.log_alias_to_log_qnames.values() for qname in qnames}

    def get_artifact_log_qnames(self):
        if False:
            print('Hello World!')
        return set(self.artifact_log_qnames)

    def get_child_log_qnames(self):
        if False:
            return 10
        return set(self.child_log_qnames)

    def is_off_by_default(self, artifact_qname):
        if False:
            return 10
        return artifact_qname in self.off_by_default_artifact_names

@dataclass
class LogState:
    log_qname_to_level: Dict[str, str] = field(default_factory=dict)
    artifact_names: Set[str] = field(default_factory=set)

    def enable_artifact(self, artifact_name):
        if False:
            i = 10
            return i + 15
        self.artifact_names.add(artifact_name)

    def is_artifact_enabled(self, name):
        if False:
            while True:
                i = 10
        return name in self.artifact_names

    def enable_log(self, log_qnames, log_level):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(log_qnames, str):
            log_qnames = [log_qnames]
        for log_qname in log_qnames:
            self.log_qname_to_level[log_qname] = log_level

    def get_log_level_pairs(self):
        if False:
            i = 10
            return i + 15
        'Returns all qualified module names for which the user requested\n        explicit logging settings.\n\n        .. warning:\n\n            This function used to return all loggers, regardless of whether\n            or not the user specified them or not; it now only returns logs\n            which were explicitly mentioned by the user (and torch, which\n            always is implicitly requested when we initialize our logging\n            subsystem.)\n        '
        return self.log_qname_to_level.items()

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        self.log_qname_to_level.clear()
        self.artifact_names.clear()
log_registry = LogRegistry()
log_state = LogState()
DEFAULT_LOGGING = {'dynamo': logging.INFO, 'graph_code': True, 'aot': logging.INFO, 'graph_breaks': True, 'recompiles': True, 'dynamic': logging.INFO, 'guards': True, 'trace_source': True}

def set_logs(*, all: int=DEFAULT_LOG_LEVEL, dynamo: Optional[int]=None, aot: Optional[int]=None, dynamic: Optional[int]=None, inductor: Optional[int]=None, distributed: Optional[int]=None, onnx: Optional[int]=None, bytecode: bool=False, aot_graphs: bool=False, aot_joint_graph: bool=False, ddp_graphs: bool=False, graph: bool=False, graph_code: bool=False, graph_breaks: bool=False, graph_sizes: bool=False, guards: bool=False, recompiles: bool=False, trace_source: bool=False, trace_call: bool=False, output_code: bool=False, schedule: bool=False, perf_hints: bool=False, post_grad_graphs: bool=False, onnx_diagnostics: bool=False, fusion: bool=False, overlap: bool=False, modules: Optional[Dict[str, Union[int, bool]]]=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Sets the log level for individual components and toggles individual log\n    artifact types.\n\n    .. warning:: This feature is a prototype and may have compatibility\n        breaking changes in the future.\n\n    .. note:: The ``TORCH_LOGS`` environment variable has complete precedence\n        over this function, so if it was set, this function does nothing.\n\n    A component is a set of related features in PyTorch. All of the log\n    messages emitted from a given component have their own log levels. If the\n    log level of a particular message has priority greater than or equal to its\n    component\'s log level setting, it is emitted. Otherwise, it is supressed.\n    This allows you to, for instance, silence large groups of log messages that\n    are not relevant to you and increase verbosity of logs for components that\n    are relevant. The expected log level values, ordered from highest to lowest\n    priority, are:\n\n        * ``logging.CRITICAL``\n        * ``logging.ERROR``\n        * ``logging.WARNING``\n        * ``logging.INFO``\n        * ``logging.DEBUG``\n        * ``logging.NOTSET``\n\n    See documentation for the Python ``logging`` module for more information on\n    log levels: `<https://docs.python.org/3/library/logging.html#logging-levels>`_\n\n    An artifact is a particular type of log message. Each artifact is assigned\n    to a parent component. A component can emit many different kinds of\n    artifacts. In general, an artifact is emitted if either its corresponding\n    setting in the argument list below is turned on or if its parent component\n    is set to a log level less than or equal to the log level of the artifact.\n\n    Keyword args:\n        all (:class:`Optional[int]`):\n            The default log level for all components. Default: ``logging.WARN``\n\n        dynamo (:class:`Optional[int]`):\n            The log level for the TorchDynamo component. Default: ``logging.WARN``\n\n        aot (:class:`Optional[int]`):\n            The log level for the AOTAutograd component. Default: ``logging.WARN``\n\n        inductor (:class:`Optional[int]`):\n            The log level for the TorchInductor component. Default: ``logging.WARN``\n\n        dynamic (:class:`Optional[int]`):\n            The log level for dynamic shapes. Default: ``logging.WARN``\n\n        distributed (:class:`Optional[int]`):\n            Whether to log communication operations and other debug info from pytorch distributed components.\n            Default: ``logging.WARN``\n\n        onnx (:class:`Optional[int]`):\n            The log level for the ONNX exporter component. Default: ``logging.WARN``\n\n        bytecode (:class:`bool`):\n            Whether to emit the original and generated bytecode from TorchDynamo.\n            Default: ``False``\n\n        aot_graphs (:class:`bool`):\n            Whether to emit the graphs generated by AOTAutograd. Default: ``False``\n\n        aot_joint_graph (:class:`bool`):\n            Whether to emit the joint forward-backward graph generated by AOTAutograd. Default: ``False``\n\n        ddp_graphs (:class:`bool`):\n            Whether to emit graphs generated by DDPOptimizer. Default: ``False``\n\n        graph (:class:`bool`):\n            Whether to emit the graph captured by TorchDynamo in tabular format.\n            Default: ``False``\n\n        graph_code (:class:`bool`):\n            Whether to emit the python source of the graph captured by TorchDynamo.\n            Default: ``False``\n\n        graph_breaks (:class:`bool`):\n            Whether to emit the graph breaks encountered by TorchDynamo.\n            Default: ``False``\n\n        graph_sizes (:class:`bool`):\n            Whether to emit tensor sizes of the graph captured by TorchDynamo.\n            Default: ``False``\n\n        guards (:class:`bool`):\n            Whether to emit the guards generated by TorchDynamo for each compiled\n            function. Default: ``False``\n\n        recompiles (:class:`bool`):\n            Whether to emit a guard failure reason and message every time\n            TorchDynamo recompiles a function. Default: ``False``\n\n        trace_source (:class:`bool`):\n            Whether to emit when TorchDynamo begins tracing a new line. Default: ``False``\n\n        trace_call (:class:`bool`):\n            Whether to emit detailed line location when TorchDynamo creates an FX node\n            corresponding to function call. Python 3.11+ only. Default: ``False``\n\n        output_code (:class:`bool`):\n            Whether to emit the TorchInductor output code. Default: ``False``\n\n        schedule (:class:`bool`):\n            Whether to emit the TorchInductor schedule. Default: ``False``\n\n        perf_hints (:class:`bool`):\n            Whether to emit the TorchInductor perf hints. Default: ``False``\n\n        post_grad_graphs (:class:`bool`):\n            Whether to emit the graphs generated by after post grad passes. Default: ``False``\n\n        onnx_diagnostics (:class:`bool`):\n            Whether to emit the ONNX exporter diagnostics in logging. Default: ``False``\n\n        fusion (:class:`bool`):\n            Whether to emit detailed Inductor fusion decisions. Default: ``False``\n\n        overlap (:class:`bool`):\n            Whether to emit detailed Inductor compute/comm overlap decisions. Default: ``False``\n\n        modules (dict):\n            This argument provides an alternate way to specify the above log\n            component and artifact settings, in the format of a keyword args\n            dictionary given as a single argument. There are two cases\n            where this is useful (1) if a new log component or artifact has\n            been registered but a keyword argument for it has not been added\n            to this function and (2) if the log level for an unregistered module\n            needs to be set. This can be done by providing the fully-qualified module\n            name as the key, with the log level as the value. Default: ``None``\n\n\n    Example::\n\n        >>> # xdoctest: +SKIP\n        >>> import logging\n\n        # The following changes the "dynamo" component to emit DEBUG-level\n        # logs, and to emit "graph_code" artifacts.\n\n        >>> torch._logging.set_logs(dynamo=logging.DEBUG, graph_code=True)\n\n        # The following enables the logs for a different module\n\n        >>> torch._logging.set_logs(modules={"unregistered.module.name": logging.DEBUG})\n    '
    if LOG_ENV_VAR in os.environ:
        log.warning('Using TORCH_LOGS environment variable for log settings, ignoring call to set_logs')
        return
    log_state.clear()
    modules = modules or {}

    def _set_logs(**kwargs):
        if False:
            return 10
        for (alias, val) in itertools.chain(kwargs.items(), modules.items()):
            if val is None:
                continue
            if log_registry.is_artifact(alias):
                if not isinstance(val, bool):
                    raise ValueError(f'Expected bool to enable artifact {alias}, received {val}')
                if val:
                    log_state.enable_artifact(alias)
            elif log_registry.is_log(alias) or alias in log_registry.child_log_qnames:
                if val not in logging._levelToName:
                    raise ValueError(f"Unrecognized log level for log {alias}: {val}, valid level values are: {','.join([str(k) for k in logging._levelToName.keys()])}")
                log_state.enable_log(log_registry.log_alias_to_log_qnames.get(alias, alias), val)
            else:
                raise ValueError(f'Unrecognized log or artifact name passed to set_logs: {alias}')
        _init_logs()
    _set_logs(torch=all, dynamo=dynamo, aot=aot, inductor=inductor, dynamic=dynamic, bytecode=bytecode, aot_graphs=aot_graphs, aot_joint_graph=aot_joint_graph, ddp_graphs=ddp_graphs, distributed=distributed, graph=graph, graph_code=graph_code, graph_breaks=graph_breaks, graph_sizes=graph_sizes, guards=guards, recompiles=recompiles, trace_source=trace_source, trace_call=trace_call, output_code=output_code, schedule=schedule, perf_hints=perf_hints, post_grad_graphs=post_grad_graphs, onnx=onnx, onnx_diagnostics=onnx_diagnostics, fusion=fusion, overlap=overlap)

def get_loggers():
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns: a list of all registered loggers\n    '
    return [logging.getLogger(qname) for qname in log_registry.get_log_qnames()]

def register_log(setting_name, log_name):
    if False:
        return 10
    '\n    Enables a log to be controlled by the env var and user API with the setting_name\n    Args:\n        setting_name:  the shorthand name used in the env var and user API\n        log_name:  the log name that the setting_name is associated with\n    '
    log_registry.register_log(setting_name, log_name)

def register_artifact(setting_name, description, visible=False, off_by_default=False, log_format=None):
    if False:
        i = 10
        return i + 15
    '\n    Enables an artifact to be controlled by the env var and user API with name\n    Args:\n        setting_name: the shorthand name used in the env var and user API\n        description: A description of what this outputs\n        visible: Whether it gets suggested to users by default\n        off_by_default: whether this artifact should be logged when the ancestor loggers\n            are enabled at level DEBUG\n    '
    log_registry.register_artifact_name(setting_name, description, visible, off_by_default, log_format)

def getArtifactLogger(module_qname, artifact_name):
    if False:
        for i in range(10):
            print('nop')
    if artifact_name not in log_registry.artifact_names:
        raise ValueError(f'Artifact name: {repr(artifact_name)} not registered,please call register_artifact({repr(artifact_name)}) in torch._logging.registrations.')
    qname = module_qname + f'.__{artifact_name}'
    log = logging.getLogger(qname)
    log.artifact_name = artifact_name
    log_registry.register_artifact_log(qname)
    configure_artifact_log(log)
    return log
INCR_VERBOSITY_CHAR = '+'
DECR_VERBOSITY_CHAR = '-'
VERBOSITY_REGEX = '(' + '|'.join([re.escape(INCR_VERBOSITY_CHAR), re.escape(DECR_VERBOSITY_CHAR)]) + '?)'

def configure_artifact_log(log):
    if False:
        i = 10
        return i + 15
    if log_registry.is_off_by_default(log.artifact_name):
        log.propagate = False
    if log_state.is_artifact_enabled(log.artifact_name):
        log.setLevel(logging.DEBUG)
        log.propagate = True

def _gen_settings_regex():
    if False:
        return 10
    return re.compile('((\\+|-)?[\\w\\.]+,\\s*)*(\\+|-)?[\\w\\.]+?')

def _validate_settings(settings):
    if False:
        return 10
    return re.fullmatch(_gen_settings_regex(), settings) is not None

def help_message(verbose=False):
    if False:
        i = 10
        return i + 15

    def pad_to(s, length=30):
        if False:
            return 10
        assert len(s) <= length
        return s + ' ' * (length - len(s))
    if verbose:
        printed_artifacts = log_registry.artifact_names
    else:
        printed_artifacts = log_registry.visible_artifacts
    if verbose:
        heading = 'All registered names'
    else:
        heading = "Visible registered names (use TORCH_LOGS='+help' for full list)"
    lines = ['all'] + list(log_registry.log_alias_to_log_qnames.keys()) + [f'{pad_to(name)}\t{log_registry.artifact_descriptions[name]}' for name in printed_artifacts]
    setting_info = '  ' + '\n  '.join(lines)
    examples = '\nExamples:\n  TORCH_LOGS="+dynamo,aot" will set the log level of TorchDynamo to\n  logging.DEBUG and AOT to logging.INFO\n\n  TORCH_LOGS="-dynamo,+inductor" will set the log level of TorchDynamo to\n  logging.ERROR and TorchInductor to logging.DEBUG\n\n  TORCH_LOGS="aot_graphs" will enable the aot_graphs artifact\n\n  TORCH_LOGS="+dynamo,schedule" will enable set the log level of TorchDynamo\n  to logging.DEBUG and enable the schedule artifact\n\n  TORCH_LOGS="+some.random.module,schedule" will set the log level of\n  some.random.module to logging.DEBUG and enable the schedule artifact\n\n  TORCH_LOGS_FORMAT="%(levelname)s: %(message)s" or any provided format\n  string will set the output format\n  Valid keys are "levelname", "message", "pathname", "levelno", "lineno",\n  "filename" and "name".\n'
    msg = f'\nTORCH_LOGS Info\n{examples}\n\n{heading}\n{setting_info}\n'
    return msg

def _invalid_settings_err_msg(settings, verbose=False):
    if False:
        for i in range(10):
            print('nop')
    valid_settings = ', '.join(['all'] + list(log_registry.log_alias_to_log_qnames.keys()) + list(log_registry.artifact_names))
    msg = f'\nInvalid log settings: {settings}, must be a comma separated list of fully\nqualified module names, registered log names or registered artifact names.\nFor more info on various settings, try TORCH_LOGS="help"\nValid settings:\n{valid_settings}\n'
    return msg

@functools.lru_cache
def _parse_log_settings(settings):
    if False:
        while True:
            i = 10
    if settings == '':
        return dict()
    if settings == 'help':
        raise ValueError(help_message(verbose=False))
    elif settings == '+help':
        raise ValueError(help_message(verbose=True))
    if not _validate_settings(settings):
        raise ValueError(_invalid_settings_err_msg(settings))
    settings = re.sub('\\s+', '', settings)
    log_names = settings.split(',')

    def get_name_level_pair(name):
        if False:
            while True:
                i = 10
        clean_name = name.replace(INCR_VERBOSITY_CHAR, '')
        clean_name = clean_name.replace(DECR_VERBOSITY_CHAR, '')
        if name[0] == INCR_VERBOSITY_CHAR:
            level = logging.DEBUG
        elif name[0] == DECR_VERBOSITY_CHAR:
            level = logging.ERROR
        else:
            level = logging.INFO
        return (clean_name, level)
    log_state = LogState()
    for name in log_names:
        (name, level) = get_name_level_pair(name)
        if name == 'all':
            name = 'torch'
        if log_registry.is_log(name):
            assert level is not None
            log_qnames = log_registry.log_alias_to_log_qnames[name]
            log_state.enable_log(log_qnames, level)
        elif log_registry.is_artifact(name):
            log_state.enable_artifact(name)
        elif _is_valid_module(name):
            if not _has_registered_parent(name):
                log_registry.register_log(name, name)
            else:
                log_registry.register_child_log(name)
            log_state.enable_log(name, level)
        else:
            raise ValueError(_invalid_settings_err_msg(settings))
    return log_state

def _is_valid_module(qname):
    if False:
        return 10
    try:
        __import__(qname)
        return True
    except ImportError:
        return False

def _update_log_state_from_env():
    if False:
        while True:
            i = 10
    global log_state
    log_setting = os.environ.get(LOG_ENV_VAR, None)
    if log_setting is not None:
        log_state = _parse_log_settings(log_setting)

def _has_registered_parent(log_qname):
    if False:
        while True:
            i = 10
    cur_log = logging.getLogger(log_qname)
    registered_log_qnames = log_registry.get_log_qnames()
    while cur_log.parent:
        if cur_log.name in registered_log_qnames:
            return True
        cur_log = cur_log.parent
    return False

class TorchLogsFormatter(logging.Formatter):

    def format(self, record):
        if False:
            while True:
                i = 10
        artifact_name = getattr(logging.getLogger(record.name), 'artifact_name', None)
        if artifact_name is not None:
            artifact_formatter = log_registry.artifact_log_formatters.get(artifact_name, None)
            if artifact_formatter is not None:
                return artifact_formatter.format(record)
        record.message = record.getMessage()
        record.asctime = self.formatTime(record, self.datefmt)
        s = record.message
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if s[-1:] != '\n':
                s = s + '\n'
            s = s + record.exc_text
        if record.stack_info:
            if s[-1:] != '\n':
                s = s + '\n'
            s = s + self.formatStack(record.stack_info)
        lines = s.split('\n')
        record.rankprefix = ''
        if dist.is_available() and dist.is_initialized():
            record.rankprefix = f'[rank{dist.get_rank()}]:'
        record.traceid = ''
        if (trace_id := torch._guards.CompileContext.current_trace_id()) is not None:
            record.traceid = f' [{trace_id}]'
        prefix = f'{record.rankprefix}[{record.asctime}]{record.traceid} {record.name}: [{record.levelname}]'
        return '\n'.join((f'{prefix} {l}' for l in lines))

def _default_formatter():
    if False:
        while True:
            i = 10
    fmt = os.environ.get(LOG_FORMAT_ENV_VAR, None)
    if fmt is None:
        return TorchLogsFormatter()
    else:
        return logging.Formatter(fmt)
DEFAULT_FORMATTER = _default_formatter()

def _setup_handlers(create_handler_fn, log):
    if False:
        for i in range(10):
            print('nop')
    debug_handler = _track_handler(create_handler_fn())
    debug_handler.setFormatter(DEFAULT_FORMATTER)
    debug_handler.setLevel(logging.DEBUG)
    log.addHandler(debug_handler)
handlers = WeakSet()

def _track_handler(handler):
    if False:
        for i in range(10):
            print('nop')
    handlers.add(handler)
    return handler

def _is_torch_handler(handler):
    if False:
        for i in range(10):
            print('nop')
    return handler in handlers

def _clear_handlers(log):
    if False:
        print('Hello World!')
    to_remove = [handler for handler in log.handlers if _is_torch_handler(handler)]
    for handler in to_remove:
        log.removeHandler(handler)

def _reset_logs():
    if False:
        print('Hello World!')
    for log_qname in log_registry.get_log_qnames():
        log = logging.getLogger(log_qname)
        log.setLevel(logging.WARNING)
        log.propagate = False
        _clear_handlers(log)
    for artifact_log_qname in itertools.chain(log_registry.get_artifact_log_qnames(), log_registry.get_child_log_qnames()):
        log = logging.getLogger(artifact_log_qname)
        log.setLevel(logging.NOTSET)
        log.propagate = True

def _get_log_state():
    if False:
        i = 10
        return i + 15
    return log_state

def _set_log_state(state):
    if False:
        print('Hello World!')
    global log_state
    log_state = state

def _init_logs(log_file_name=None):
    if False:
        for i in range(10):
            print('nop')
    _reset_logs()
    _update_log_state_from_env()
    for log_qname in log_registry.get_log_qnames():
        log = logging.getLogger(log_qname)
        log.setLevel(logging.NOTSET)
    for (log_qname, level) in log_state.get_log_level_pairs():
        log = logging.getLogger(log_qname)
        log.setLevel(level)
    for log_qname in log_registry.get_log_qnames():
        log = logging.getLogger(log_qname)
        _setup_handlers(logging.StreamHandler, log)
        if log_file_name is not None:
            _setup_handlers(lambda : logging.FileHandler(log_file_name), log)
    for artifact_log_qname in log_registry.get_artifact_log_qnames():
        log = logging.getLogger(artifact_log_qname)
        configure_artifact_log(log)

@functools.lru_cache(None)
def warning_once(logger_obj, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    This function is similar to `logger.warning()`, but will emit the warning with the same message only once\n    Note: The cache is for the function arguments, so 2 different callers using the same arguments will hit the cache.\n    The assumption here is that all warning messages are unique across the code. If they aren't then need to switch to\n    another type of cache that includes the caller frame information in the hashing function.\n    "
    logger_obj.warning(*args, **kwargs)

class LazyString:

    def __init__(self, func, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.func(*self.args, **self.kwargs)
import torch._guards
import torch.distributed as dist