from __future__ import annotations
import itertools
import re
from typing import Any, Dict, Generator, List, Optional
import sentry_sdk
from sentry.eventstore.models import Event
from sentry.grouping.component import GroupingComponent, calculate_tree_label
from sentry.grouping.strategies.base import GroupingContext, ReturnedVariants, call_with_variants, strategy
from sentry.grouping.strategies.hierarchical import get_stacktrace_hierarchy
from sentry.grouping.strategies.message import normalize_message_for_grouping
from sentry.grouping.strategies.utils import has_url_origin, remove_non_stacktrace_variants
from sentry.grouping.utils import hash_from_values
from sentry.interfaces.exception import Exception as ChainedException
from sentry.interfaces.exception import Mechanism, SingleException
from sentry.interfaces.stacktrace import Frame, Stacktrace
from sentry.interfaces.threads import Threads
from sentry.stacktraces.platform import get_behavior_family_for_platform
_ruby_erb_func = re.compile('__\\d{4,}_\\d{4,}$')
_basename_re = re.compile('[/\\\\]')
_java_reflect_enhancer_re = re.compile('(sun\\.reflect\\.Generated(?:Serialization)?ConstructorAccessor)\\d+', re.X)
_java_cglib_enhancer_re = re.compile('(\\$\\$[\\w_]+?CGLIB\\$\\$)[a-fA-F0-9]+(_[0-9]+)?', re.X)
_java_assist_enhancer_re = re.compile('(\\$\\$_javassist)(?:_seam)?(?:_[0-9]+)?', re.X)
_clojure_enhancer_re = re.compile('(\\$fn__)\\d+', re.X)
_java_enhancer_by_re = re.compile('(\\$\\$?EnhancerBy[\\w_]+?\\$?\\$)[a-fA-F0-9]+(_[0-9]+)?', re.X)
_java_fast_class_by_re = re.compile('(\\$\\$?FastClassBy[\\w_]+?\\$?\\$)[a-fA-F0-9]+(_[0-9]+)?', re.X)
_java_hibernate_proxy_re = re.compile('(\\$\\$?HibernateProxy\\$?\\$)\\w+(_[0-9]+)?', re.X)
RECURSION_COMPARISON_FIELDS = ['abs_path', 'package', 'module', 'filename', 'function', 'lineno', 'colno']
StacktraceEncoderReturnValue = Any

def is_recursion_v1(frame1: Frame, frame2: Frame | None) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a boolean indicating whether frames are recursive calls.\n    '
    if frame2 is None:
        return False
    for field in RECURSION_COMPARISON_FIELDS:
        if getattr(frame1, field, None) != getattr(frame2, field, None):
            return False
    return True

def get_basename(string: str) -> str:
    if False:
        return 10
    '\n    Returns best-effort basename of a string irrespective of platform.\n    '
    return _basename_re.split(string)[-1]

def get_package_component(package: str, platform: Optional[str]) -> GroupingComponent:
    if False:
        for i in range(10):
            print('nop')
    if package is None or platform != 'native':
        return GroupingComponent(id='package')
    package = get_basename(package).lower()
    package_component = GroupingComponent(id='package', values=[package])
    return package_component

def get_filename_component(abs_path: str, filename: Optional[str], platform: Optional[str], allow_file_origin: bool=False) -> GroupingComponent:
    if False:
        return 10
    'Attempt to normalize filenames by detecting special filenames and by\n    using the basename only.\n    '
    if filename is None:
        return GroupingComponent(id='filename')
    filename = _basename_re.split(filename)[-1].lower()
    filename_component = GroupingComponent(id='filename', values=[filename])
    if has_url_origin(abs_path, allow_file_origin=allow_file_origin):
        filename_component.update(contributes=False, hint='ignored because frame points to a URL')
    elif filename == '<anonymous>':
        filename_component.update(contributes=False, hint='anonymous filename discarded')
    elif filename == '[native code]':
        filename_component.update(contributes=False, hint='native code indicated by filename')
    elif platform == 'java':
        new_filename = _java_assist_enhancer_re.sub('\\1<auto>', filename)
        if new_filename != filename:
            filename_component.update(values=[new_filename], hint='cleaned javassist parts')
            filename = new_filename
    filename_component.update(tree_label={'filebase': get_basename(filename)})
    return filename_component

def get_module_component(abs_path: Optional[str], module: Optional[str], platform: Optional[str], context: GroupingContext) -> GroupingComponent:
    if False:
        while True:
            i = 10
    'Given an absolute path, module and platform returns the module component\n    with some necessary cleaning performed.\n    '
    if module is None:
        return GroupingComponent(id='module')
    module_component = GroupingComponent(id='module', values=[module])
    if platform == 'javascript' and '/' in module and abs_path and abs_path.endswith(module):
        module_component.update(contributes=False, hint='ignored bad javascript module')
    elif platform == 'java':
        if '$$Lambda$' in module:
            module_component.update(contributes=False, hint='ignored java lambda')
        if module[:35] == 'sun.reflect.GeneratedMethodAccessor':
            module_component.update(values=['sun.reflect.GeneratedMethodAccessor'], hint='removed reflection marker')
        elif module[:44] == 'jdk.internal.reflect.GeneratedMethodAccessor':
            module_component.update(values=['jdk.internal.reflect.GeneratedMethodAccessor'], hint='removed reflection marker')
        else:
            old_module = module
            module = _java_reflect_enhancer_re.sub('\\1<auto>', module)
            module = _java_cglib_enhancer_re.sub('\\1<auto>', module)
            module = _java_assist_enhancer_re.sub('\\1<auto>', module)
            module = _clojure_enhancer_re.sub('\\1<auto>', module)
            if context['java_cglib_hibernate_logic']:
                module = _java_enhancer_by_re.sub('\\1<auto>', module)
                module = _java_fast_class_by_re.sub('\\1<auto>', module)
                module = _java_hibernate_proxy_re.sub('\\1<auto>', module)
            if module != old_module:
                module_component.update(values=[module], hint='removed codegen marker')
        for part in reversed(module.split('.')):
            if '$' not in part:
                module_component.update(tree_label={'classbase': part})
                break
    return module_component

def get_function_component(context: GroupingContext, function: Optional[str], raw_function: Optional[str], platform: Optional[str], sourcemap_used: bool=False, context_line_available: bool=False) -> GroupingComponent:
    if False:
        i = 10
        return i + 15
    "\n    Attempt to normalize functions by removing common platform outliers.\n\n    - Ruby generates (random?) integers for various anonymous style functions\n      such as in erb and the active_support library.\n    - Block functions have metadata that we don't care about.\n\n    The `legacy_function_logic` parameter controls if the system should\n    use the frame v1 function name logic or the frame v2 logic.  The difference\n    is that v2 uses the function name consistently and v1 prefers raw function\n    or a trimmed version (of the truncated one) for native.  Related to this is\n    the `prefer_raw_function_name` flag which just flat out prefers the\n    raw function name over the non raw one.\n    "
    from sentry.stacktraces.functions import trim_function_name
    behavior_family = get_behavior_family_for_platform(platform)
    prefer_raw_function_name = platform == 'csharp'
    if context['legacy_function_logic'] or prefer_raw_function_name:
        func = raw_function or function
    else:
        func = function or raw_function
        if not raw_function and function:
            func = trim_function_name(func, platform)
    if not func:
        return GroupingComponent(id='function')
    function_component = GroupingComponent(id='function', values=[func])
    if platform == 'ruby':
        if func.startswith('block '):
            function_component.update(values=['block'], hint='ruby block')
        else:
            new_function = _ruby_erb_func.sub('', func)
            if new_function != func:
                function_component.update(values=[new_function], hint='removed generated erb template suffix')
    elif platform == 'php':
        if func.startswith(('[Anonymous', 'class@anonymous\x00')):
            function_component.update(contributes=False, hint='ignored anonymous function')
        if context['php_detect_anonymous_classes'] and func.startswith('class@anonymous'):
            new_function = func.rsplit('::', 1)[-1]
            if new_function != func:
                function_component.update(values=[new_function], hint='anonymous class method')
    elif platform == 'java':
        if func.startswith('lambda$'):
            function_component.update(contributes=False, hint='ignored lambda function')
    elif behavior_family == 'native':
        if func in ('<redacted>', '<unknown>'):
            function_component.update(contributes=False, hint='ignored unknown function')
        elif context['legacy_function_logic']:
            new_function = trim_function_name(func, platform, normalize_lambdas=False)
            if new_function != func:
                function_component.update(values=[new_function], hint='isolated function')
                func = new_function
        if context['native_fuzzing']:
            new_function = func.replace('(anonymous namespace)', "`anonymous namespace'")
            if new_function != func:
                function_component.update(values=[new_function])
    elif context['javascript_fuzzing'] and behavior_family == 'javascript':
        new_function = func.rsplit('.', 1)[-1]
        if new_function != func:
            function_component.update(values=[new_function], hint='trimmed javascript function')
        if sourcemap_used and context_line_available:
            function_component.update(contributes=False, hint='ignored because sourcemap used and context line available')
    if function_component.values and context['hierarchical_grouping']:
        function_component.update(tree_label={'function': function_component.values[0]})
    return function_component

@strategy(ids=['frame:v1'], interface=Frame)
def frame(interface: Frame, event: Event, context: GroupingContext, **meta: Any) -> ReturnedVariants:
    if False:
        for i in range(10):
            print('nop')
    frame = interface
    platform = frame.platform or event.platform
    filename_component = get_filename_component(frame.abs_path, frame.filename, platform, allow_file_origin=context['javascript_fuzzing'])
    module_component = get_module_component(frame.abs_path, frame.module, platform, context)
    if module_component.contributes and filename_component.contributes:
        filename_component.update(contributes=False, hint='module takes precedence')
    context_line_component = None
    if platform in context['contextline_platforms']:
        context_line_component = get_contextline_component(frame, platform, function=frame.function, context=context)
    context_line_available = bool(context_line_component and context_line_component.contributes)
    function_component = get_function_component(context=context, function=frame.function, raw_function=frame.raw_function, platform=platform, sourcemap_used=frame.data and frame.data.get('sourcemap') is not None, context_line_available=context_line_available)
    values = [module_component, filename_component, function_component]
    if context_line_component is not None:
        context_line_component.update(tree_label=function_component.tree_label)
        values.append(context_line_component)
    if context['discard_native_filename'] and get_behavior_family_for_platform(platform) == 'native' and function_component.contributes and filename_component.contributes:
        filename_component.update(contributes=False, hint='discarded native filename for grouping stability')
    if context['use_package_fallback'] and frame.package:
        package_component = get_package_component(package=frame.package, platform=platform)
        if package_component.contributes:
            use_package_component = all((not component.contributes for component in values))
            if use_package_component:
                package_component.update(hint='used as fallback because function name is not available')
            else:
                package_component.update(contributes=False, hint='ignored because function takes precedence')
            if package_component.values and context['hierarchical_grouping']:
                package_component.update(tree_label={'package': package_component.values[0]})
            values.append(package_component)
    rv = GroupingComponent(id='frame', values=values)
    if context['javascript_fuzzing'] and get_behavior_family_for_platform(platform) == 'javascript':
        func = frame.raw_function or frame.function
        if func:
            func = func.rsplit('.', 1)[-1]
        if not func:
            function_component.update(contributes=False)
        elif func in ('?', '<anonymous function>', '<anonymous>', 'Anonymous function') or func.endswith('/<'):
            function_component.update(contributes=False, hint='ignored unknown function name')
        if func == 'eval' or frame.abs_path in ('[native code]', 'native code', 'eval code', '<anonymous>'):
            rv.update(contributes=False, hint='ignored low quality javascript frame')
    if context['is_recursion']:
        rv.update(contributes=False, hint='ignored due to recursion')
    if rv.contributes:
        tree_label = {}
        for value in rv.values:
            if isinstance(value, GroupingComponent) and value.contributes and value.tree_label:
                tree_label.update(value.tree_label)
        if tree_label and context['hierarchical_grouping']:
            tree_label['datapath'] = frame.datapath
            rv.tree_label = tree_label
        else:
            rv.tree_label = None
    return {context['variant']: rv}

def get_contextline_component(frame: Frame, platform: Optional[str], function: str, context: GroupingContext) -> GroupingComponent:
    if False:
        for i in range(10):
            print('nop')
    "Returns a contextline component.  The caller's responsibility is to\n    make sure context lines are only used for platforms where we trust the\n    quality of the sourcecode.  It does however protect against some bad\n    JavaScript environments based on origin checks.\n    "
    line = ' '.join((frame.context_line or '').expandtabs(2).split())
    if not line:
        return GroupingComponent(id='context-line')
    component = GroupingComponent(id='context-line', values=[line])
    if line:
        if len(frame.context_line) > 120:
            component.update(hint='discarded because line too long', contributes=False)
        elif get_behavior_family_for_platform(platform) == 'javascript':
            if context['with_context_line_file_origin_bug']:
                if has_url_origin(frame.abs_path, allow_file_origin=True):
                    component.update(hint='discarded because from URL origin', contributes=False)
            elif not function and has_url_origin(frame.abs_path):
                component.update(hint='discarded because from URL origin and no function', contributes=False)
    return component

@strategy(ids=['stacktrace:v1'], interface=Stacktrace, score=1800)
def stacktrace(interface: Stacktrace, event: Event, context: GroupingContext, **meta: Any) -> ReturnedVariants:
    if False:
        while True:
            i = 10
    assert context['variant'] is None
    if context['hierarchical_grouping']:
        with context:
            context['variant'] = 'system'
            return _single_stacktrace_variant(interface, event=event, context=context, meta=meta)
    else:
        return call_with_variants(_single_stacktrace_variant, ['!system', 'app'], interface, event=event, context=context, meta=meta)

def _single_stacktrace_variant(stacktrace: Stacktrace, event: Event, context: GroupingContext, meta: Dict[str, Any]) -> ReturnedVariants:
    if False:
        for i in range(10):
            print('nop')
    variant = context['variant']
    frames = stacktrace.frames
    values: List[GroupingComponent] = []
    prev_frame = None
    frames_for_filtering = []
    for frame in frames:
        with context:
            context['is_recursion'] = is_recursion_v1(frame, prev_frame)
            frame_component = context.get_grouping_component(frame, event=event, **meta)
        if not context['hierarchical_grouping'] and variant == 'app' and (not frame.in_app):
            frame_component.update(contributes=False, hint='non app frame')
        values.append(frame_component)
        frames_for_filtering.append(frame.get_raw_data())
        prev_frame = frame
    if len(frames) == 1 and values[0].contributes and (get_behavior_family_for_platform(frames[0].platform or event.platform) == 'javascript') and (not frames[0].function) and frames[0].is_url():
        values[0].update(contributes=False, hint='ignored single non-URL JavaScript frame')
    (main_variant, inverted_hierarchy) = context.config.enhancements.assemble_stacktrace_component(values, frames_for_filtering, event.platform, exception_data=context['exception_data'])
    if inverted_hierarchy is None:
        inverted_hierarchy = stacktrace.snapshot
    inverted_hierarchy = bool(inverted_hierarchy)
    if not context['hierarchical_grouping']:
        return {variant: main_variant}
    all_variants: ReturnedVariants = get_stacktrace_hierarchy(main_variant, values, frames_for_filtering, inverted_hierarchy)
    all_variants['system'] = main_variant
    return all_variants

@stacktrace.variant_processor
def stacktrace_variant_processor(variants: ReturnedVariants, context: GroupingContext, **meta: Any) -> ReturnedVariants:
    if False:
        i = 10
        return i + 15
    return remove_non_stacktrace_variants(variants)

@strategy(ids=['single-exception:v1'], interface=SingleException)
def single_exception(interface: SingleException, event: Event, context: GroupingContext, **meta: Any) -> ReturnedVariants:
    if False:
        i = 10
        return i + 15
    type_component = GroupingComponent(id='type', values=[interface.type] if interface.type else [])
    system_type_component = type_component.shallow_copy()
    ns_error_component = None
    if interface.mechanism:
        if interface.mechanism.synthetic:
            type_component.update(contributes=False, hint='ignored because exception is synthetic')
        if interface.mechanism.meta and 'ns_error' in interface.mechanism.meta:
            ns_error_component = GroupingComponent(id='ns-error', values=[interface.mechanism.meta['ns_error'].get('domain'), interface.mechanism.meta['ns_error'].get('code')])
    if interface.stacktrace is not None:
        with context:
            context['exception_data'] = interface.to_json()
            stacktrace_variants = context.get_grouping_component(interface.stacktrace, event=event, **meta)
    else:
        stacktrace_variants = {'app': GroupingComponent(id='stacktrace')}
    rv = {}
    for (variant, stacktrace_component) in stacktrace_variants.items():
        values = [stacktrace_component, system_type_component if variant == 'system' else type_component]
        if ns_error_component is not None:
            values.append(ns_error_component)
        if context['with_exception_value_fallback']:
            value_component = GroupingComponent(id='value')
            raw = interface.value
            if raw is not None:
                normalized = normalize_message_for_grouping(raw)
                hint = 'stripped event-specific values' if raw != normalized else None
                if normalized:
                    value_component.update(values=[normalized], hint=hint)
            if stacktrace_component.contributes and value_component.contributes:
                value_component.update(contributes=False, hint='ignored because stacktrace takes precedence')
            if ns_error_component is not None and ns_error_component.contributes and value_component.contributes:
                value_component.update(contributes=False, hint='ignored because ns-error info takes precedence')
            values.append(value_component)
        rv[variant] = GroupingComponent(id='exception', values=values)
    return rv

@strategy(ids=['chained-exception:v1'], interface=ChainedException, score=2000)
def chained_exception(interface: ChainedException, event: Event, context: GroupingContext, **meta: Any) -> ReturnedVariants:
    if False:
        for i in range(10):
            print('nop')
    all_exceptions = interface.exceptions()
    exception_components = {id(exception): context.get_grouping_component(exception, event=event, **meta) for exception in all_exceptions}
    with sentry_sdk.start_span(op='grouping.strategies.newstyle.filter_exceptions_for_exception_groups') as span:
        try:
            exceptions = filter_exceptions_for_exception_groups(all_exceptions, exception_components, event)
        except Exception:
            sentry_sdk.capture_exception()
            span.set_status('internal_error')
            exceptions = all_exceptions
    if len(exceptions) == 1:
        return exception_components[id(exceptions[0])]
    by_name: Dict[str, List[GroupingComponent]] = {}
    for exception in exceptions:
        for (name, component) in exception_components[id(exception)].items():
            by_name.setdefault(name, []).append(component)
    rv = {}
    for (name, component_list) in by_name.items():
        rv[name] = GroupingComponent(id='chained-exception', values=component_list, tree_label=calculate_tree_label(reversed(component_list)))
    return rv

def filter_exceptions_for_exception_groups(exceptions: List[SingleException], exception_components: Dict[int, GroupingComponent], event: Event) -> List[SingleException]:
    if False:
        i = 10
        return i + 15
    if len(exceptions) <= 1:
        return exceptions

    class ExceptionTreeNode:

        def __init__(self, exception: Optional[SingleException]=None, children: Optional[List[SingleException]]=None):
            if False:
                i = 10
                return i + 15
            self.exception = exception
            self.children = children if children else []
    exception_tree: Dict[int, ExceptionTreeNode] = {}
    for exception in reversed(exceptions):
        mechanism: Mechanism = exception.mechanism
        if mechanism and mechanism.exception_id is not None:
            node = exception_tree.setdefault(mechanism.exception_id, ExceptionTreeNode()).exception = exception
            node.exception = exception
            if mechanism.parent_id is not None:
                parent_node = exception_tree.setdefault(mechanism.parent_id, ExceptionTreeNode())
                parent_node.children.append(exception)
        else:
            return exceptions

    def get_child_exceptions(exception: SingleException) -> List[SingleException]:
        if False:
            return 10
        exception_id = exception.mechanism.exception_id
        node = exception_tree.get(exception_id)
        return node.children if node else []

    def get_top_level_exceptions(exception: SingleException) -> Generator[SingleException, None, None]:
        if False:
            for i in range(10):
                print('nop')
        if exception.mechanism.is_exception_group:
            children = get_child_exceptions(exception)
            yield from itertools.chain.from_iterable((get_top_level_exceptions(child) for child in children))
        else:
            yield exception

    def get_first_path(exception: SingleException) -> Generator[SingleException, None, None]:
        if False:
            for i in range(10):
                print('nop')
        yield exception
        children = get_child_exceptions(exception)
        if children:
            yield from get_first_path(children[0])
    top_level_exceptions = sorted(get_top_level_exceptions(exception_tree[0].exception), key=lambda exception: str(exception.type), reverse=True)
    distinct_top_level_exceptions = [next(group) for (_, group) in itertools.groupby(top_level_exceptions, key=lambda exception: hash_from_values(exception_components[id(exception)].values()) or '')]
    if len(distinct_top_level_exceptions) == 1:
        main_exception = distinct_top_level_exceptions[0]
        event.data['main_exception_id'] = main_exception.mechanism.exception_id
        return list(get_first_path(main_exception))
    distinct_top_level_exceptions.append(exception_tree[0].exception)
    return distinct_top_level_exceptions

@chained_exception.variant_processor
def chained_exception_variant_processor(variants: ReturnedVariants, context: GroupingContext, **meta: Any) -> ReturnedVariants:
    if False:
        print('Hello World!')
    return remove_non_stacktrace_variants(variants)

@strategy(ids=['threads:v1'], interface=Threads, score=1900)
def threads(interface: Threads, event: Event, context: GroupingContext, **meta: Any) -> ReturnedVariants:
    if False:
        return 10
    thread_variants = _filtered_threads([thread for thread in interface.values if thread.get('crashed')], event, context, meta)
    if thread_variants is not None:
        return thread_variants
    thread_variants = _filtered_threads([thread for thread in interface.values if thread.get('current')], event, context, meta)
    if thread_variants is not None:
        return thread_variants
    thread_variants = _filtered_threads(interface.values, event, context, meta)
    if thread_variants is not None:
        return thread_variants
    return {'app': GroupingComponent(id='threads', contributes=False, hint='ignored because does not contain exactly one crashing, one current or just one thread, instead contains %s threads' % len(interface.values))}

def _filtered_threads(threads: List[Dict[str, Any]], event: Event, context: GroupingContext, meta: Dict[str, Any]) -> Optional[ReturnedVariants]:
    if False:
        for i in range(10):
            print('nop')
    if len(threads) != 1:
        return None
    stacktrace = threads[0].get('stacktrace')
    if not stacktrace:
        return {'app': GroupingComponent(id='threads', contributes=False, hint='thread has no stacktrace')}
    rv = {}
    for (name, stacktrace_component) in context.get_grouping_component(stacktrace, event=event, **meta).items():
        rv[name] = GroupingComponent(id='threads', values=[stacktrace_component])
    return rv

@threads.variant_processor
def threads_variant_processor(variants: ReturnedVariants, context: GroupingContext, **meta: Any) -> ReturnedVariants:
    if False:
        i = 10
        return i + 15
    return remove_non_stacktrace_variants(variants)