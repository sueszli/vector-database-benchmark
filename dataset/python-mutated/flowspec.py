import inspect
import os
import sys
import traceback
from itertools import islice
from types import FunctionType, MethodType
from typing import Any, Callable, List, Optional, Tuple
from . import cmd_with_io
from .parameters import DelayedEvaluationParameter, Parameter
from .exception import MetaflowException, MissingInMergeArtifactsException, UnhandledInMergeArtifactsException
from .graph import FlowGraph
from .unbounded_foreach import UnboundedForeachInput
try:
    basestring
except NameError:
    basestring = str
from .datastore.inputs import Inputs

class InvalidNextException(MetaflowException):
    headline = 'Invalid self.next() transition detected'

    def __init__(self, msg):
        if False:
            print('Hello World!')
        (_, line_no, _, _) = traceback.extract_stack()[-3]
        super(InvalidNextException, self).__init__(msg, line_no)

class ParallelUBF(UnboundedForeachInput):
    """
    Unbounded-for-each placeholder for supporting parallel (multi-node) steps.
    """

    def __init__(self, num_parallel):
        if False:
            for i in range(10):
                print('nop')
        self.num_parallel = num_parallel

    def __getitem__(self, item):
        if False:
            while True:
                i = 10
        return item or 0

class FlowSpec(object):
    """
    Main class from which all Flows should inherit.

    Attributes
    ----------
    index
    input
    """
    _EPHEMERAL = {'_EPHEMERAL', '_NON_PARAMETERS', '_datastore', '_cached_input', '_graph', '_flow_decorators', '_steps', 'index', 'input'}
    _NON_PARAMETERS = {'cmd', 'foreach_stack', 'index', 'input', 'script_name', 'name'}
    _flow_decorators = {}

    def __init__(self, use_cli=True):
        if False:
            i = 10
            return i + 15
        '\n        Construct a FlowSpec\n\n        Parameters\n        ----------\n        use_cli : bool, default: True\n            Set to True if the flow is invoked from __main__ or the command line\n        '
        self.name = self.__class__.__name__
        self._datastore = None
        self._transition = None
        self._cached_input = {}
        self._graph = FlowGraph(self.__class__)
        self._steps = [getattr(self, node.name) for node in self._graph]
        if use_cli:
            from . import cli
            cli.main(self)

    @property
    def script_name(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        [Legacy function - do not use. Use `current` instead]\n\n        Returns the name of the script containing the flow\n\n        Returns\n        -------\n        str\n            A string containing the name of the script\n        '
        fname = inspect.getfile(self.__class__)
        if fname.endswith('.pyc'):
            fname = fname[:-1]
        return os.path.basename(fname)

    def _set_constants(self, graph, kwargs):
        if False:
            while True:
                i = 10
        from metaflow.decorators import flow_decorators
        seen = set()
        for (var, param) in self._get_parameters():
            norm = param.name.lower()
            if norm in seen:
                raise MetaflowException('Parameter *%s* is specified twice. Note that parameter names are case-insensitive.' % param.name)
            seen.add(norm)
        seen.clear()
        self._success = True
        parameters_info = []
        for (var, param) in self._get_parameters():
            seen.add(var)
            val = kwargs[param.name.replace('-', '_').lower()]
            if isinstance(val, DelayedEvaluationParameter):
                val = val()
            val = val.split(param.separator) if val and param.separator else val
            setattr(self, var, val)
            parameters_info.append({'name': var, 'type': param.__class__.__name__})
        constants_info = []
        for var in dir(self.__class__):
            if var[0] == '_' or var in self._NON_PARAMETERS or var in seen:
                continue
            val = getattr(self.__class__, var)
            if isinstance(val, (MethodType, FunctionType, property, type)):
                continue
            constants_info.append({'name': var, 'type': type(val).__name__})
            setattr(self, var, val)
        (steps_info, graph_structure) = graph.output_steps()
        graph_info = {'file': os.path.basename(os.path.abspath(sys.argv[0])), 'parameters': parameters_info, 'constants': constants_info, 'steps': steps_info, 'graph_structure': graph_structure, 'doc': graph.doc, 'decorators': [{'name': deco.name, 'attributes': deco.attributes, 'statically_defined': deco.statically_defined} for deco in flow_decorators() if not deco.name.startswith('_')]}
        self._graph_info = graph_info

    def _get_parameters(self):
        if False:
            while True:
                i = 10
        for var in dir(self):
            if var[0] == '_' or var in self._NON_PARAMETERS:
                continue
            try:
                val = getattr(self, var)
            except:
                continue
            if isinstance(val, Parameter):
                yield (var, val)

    def _set_datastore(self, datastore):
        if False:
            for i in range(10):
                print('nop')
        self._datastore = datastore

    def __iter__(self):
        if False:
            print('Hello World!')
        '\n        [Legacy function - do not use]\n\n        Iterate over all steps in the Flow\n\n        Returns\n        -------\n        Iterator[graph.DAGNode]\n            Iterator over the steps in the flow\n        '
        return iter(self._steps)

    def __getattr__(self, name: str):
        if False:
            print('Hello World!')
        if self._datastore and name in self._datastore:
            x = self._datastore[name]
            setattr(self, name, x)
            return x
        else:
            raise AttributeError("Flow %s has no attribute '%s'" % (self.name, name))

    def cmd(self, cmdline, input={}, output=[]):
        if False:
            return 10
        '\n        [Legacy function - do not use]\n        '
        return cmd_with_io.cmd(cmdline, input=input, output=output)

    @property
    def index(self) -> Optional[int]:
        if False:
            i = 10
            return i + 15
        '\n        The index of this foreach branch.\n\n        In a foreach step, multiple instances of this step (tasks) will be executed,\n        one for each element in the foreach. This property returns the zero based index\n        of the current task. If this is not a foreach step, this returns None.\n\n        If you need to know the indices of the parent tasks in a nested foreach, use\n        `FlowSpec.foreach_stack`.\n\n        Returns\n        -------\n        int, optional\n            Index of the task in a foreach step.\n        '
        if self._foreach_stack:
            return self._foreach_stack[-1].index

    @property
    def input(self) -> Optional[Any]:
        if False:
            return 10
        '\n        The value of the foreach artifact in this foreach branch.\n\n        In a foreach step, multiple instances of this step (tasks) will be executed,\n        one for each element in the foreach. This property returns the element passed\n        to the current task. If this is not a foreach step, this returns None.\n\n        If you need to know the values of the parent tasks in a nested foreach, use\n        `FlowSpec.foreach_stack`.\n\n        Returns\n        -------\n        object, optional\n            Input passed to the foreach task.\n        '
        return self._find_input()

    def foreach_stack(self) -> Optional[List[Tuple[int, int, Any]]]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns the current stack of foreach indexes and values for the current step.\n\n        Use this information to understand what data is being processed in the current\n        foreach branch. For example, considering the following code:\n        ```\n        @step\n        def root(self):\n            self.split_1 = ['a', 'b', 'c']\n            self.next(self.nest_1, foreach='split_1')\n\n        @step\n        def nest_1(self):\n            self.split_2 = ['d', 'e', 'f', 'g']\n            self.next(self.nest_2, foreach='split_2'):\n\n        @step\n        def nest_2(self):\n            foo = self.foreach_stack()\n        ```\n\n        `foo` will take the following values in the various tasks for nest_2:\n        ```\n            [(0, 3, 'a'), (0, 4, 'd')]\n            [(0, 3, 'a'), (1, 4, 'e')]\n            ...\n            [(0, 3, 'a'), (3, 4, 'g')]\n            [(1, 3, 'b'), (0, 4, 'd')]\n            ...\n        ```\n        where each tuple corresponds to:\n\n        - The index of the task for that level of the loop.\n        - The number of splits for that level of the loop.\n        - The value for that level of the loop.\n\n        Note that the last tuple returned in a task corresponds to:\n\n        - 1st element: value returned by `self.index`.\n        - 3rd element: value returned by `self.input`.\n\n        Returns\n        -------\n        List[Tuple[int, int, object]]\n            An array describing the current stack of foreach steps.\n        "
        return [(frame.index, frame.num_splits, self._find_input(stack_index=i)) for (i, frame) in enumerate(self._foreach_stack)]

    def _find_input(self, stack_index=None):
        if False:
            return 10
        if stack_index is None:
            stack_index = len(self._foreach_stack) - 1
        if stack_index in self._cached_input:
            return self._cached_input[stack_index]
        elif self._foreach_stack:
            frame = self._foreach_stack[stack_index]
            try:
                var = getattr(self, frame.var)
            except AttributeError:
                self._cached_input[stack_index] = None
            else:
                try:
                    self._cached_input[stack_index] = var[frame.index]
                except TypeError:
                    self._cached_input[stack_index] = next(islice(var, frame.index, frame.index + 1))
            return self._cached_input[stack_index]

    def merge_artifacts(self, inputs: Inputs, exclude: Optional[List[str]]=None, include: Optional[List[str]]=None) -> None:
        if False:
            return 10
        '\n        Helper function for merging artifacts in a join step.\n\n        This function takes all the artifacts coming from the branches of a\n        join point and assigns them to self in the calling step. Only artifacts\n        not set in the current step are considered. If, for a given artifact, different\n        values are present on the incoming edges, an error will be thrown and the artifacts\n        that conflict will be reported.\n\n        As a few examples, in the simple graph: A splitting into B and C and joining in D:\n        ```\n        A:\n          self.x = 5\n          self.y = 6\n        B:\n          self.b_var = 1\n          self.x = from_b\n        C:\n          self.x = from_c\n\n        D:\n          merge_artifacts(inputs)\n        ```\n        In D, the following artifacts are set:\n          - `y` (value: 6), `b_var` (value: 1)\n          - if `from_b` and `from_c` are the same, `x` will be accessible and have value `from_b`\n          - if `from_b` and `from_c` are different, an error will be thrown. To prevent this error,\n            you need to manually set `self.x` in D to a merged value (for example the max) prior to\n            calling `merge_artifacts`.\n\n        Parameters\n        ----------\n        inputs : Inputs\n            Incoming steps to the join point.\n        exclude : List[str], optional\n            If specified, do not consider merging artifacts with a name in `exclude`.\n            Cannot specify if `include` is also specified.\n        include : List[str], optional\n            If specified, only merge artifacts specified. Cannot specify if `exclude` is\n            also specified.\n\n        Raises\n        ------\n        MetaflowException\n            This exception is thrown if this is not called in a join step.\n        UnhandledInMergeArtifactsException\n            This exception is thrown in case of unresolved conflicts.\n        MissingInMergeArtifactsException\n            This exception is thrown in case an artifact specified in `include` cannot\n            be found.\n        '
        include = include or []
        exclude = exclude or []
        node = self._graph[self._current_step]
        if node.type != 'join':
            msg = 'merge_artifacts can only be called in a join and step *{step}* is not a join'.format(step=self._current_step)
            raise MetaflowException(msg)
        if len(exclude) > 0 and len(include) > 0:
            msg = '`exclude` and `include` are mutually exclusive in merge_artifacts'
            raise MetaflowException(msg)
        to_merge = {}
        unresolved = []
        for inp in inputs:
            if include:
                available_vars = ((var, sha) for (var, sha) in inp._datastore.items() if var in include and (not hasattr(self, var)))
            else:
                available_vars = ((var, sha) for (var, sha) in inp._datastore.items() if var not in exclude and (not hasattr(self, var)))
            for (var, sha) in available_vars:
                (_, previous_sha) = to_merge.setdefault(var, (inp, sha))
                if previous_sha != sha:
                    unresolved.append(var)
        missing = []
        for v in include:
            if v not in to_merge and (not hasattr(self, v)):
                missing.append(v)
        if unresolved:
            msg = 'Step *{step}* cannot merge the following artifacts due to them having conflicting values:\n[{artifacts}].\nTo remedy this issue, be sure to explicitly set those artifacts (using self.<artifact_name> = ...) prior to calling merge_artifacts.'.format(step=self._current_step, artifacts=', '.join(unresolved))
            raise UnhandledInMergeArtifactsException(msg, unresolved)
        if missing:
            msg = 'Step *{step}* specifies that [{include}] should be merged but [{missing}] are not present.\nTo remedy this issue, make sure that the values specified in only come from at least one branch'.format(step=self._current_step, include=', '.join(include), missing=', '.join(missing))
            raise MissingInMergeArtifactsException(msg, missing)
        for (var, (inp, _)) in to_merge.items():
            self._datastore.passdown_partial(inp._datastore, [var])

    def _validate_ubf_step(self, step_name):
        if False:
            i = 10
            return i + 15
        join_list = self._graph[step_name].out_funcs
        if len(join_list) != 1:
            msg = 'UnboundedForeach is supported only over a single node, not an arbitrary DAG. Specify a single `join` node instead of multiple:{join_list}.'.format(join_list=join_list)
            raise InvalidNextException(msg)
        join_step = join_list[0]
        join_node = self._graph[join_step]
        join_type = join_node.type
        if join_type != 'join':
            msg = "UnboundedForeach found for:{node} -> {join}. The join type isn't valid.".format(node=step_name, join=join_step)
            raise InvalidNextException(msg)

    def next(self, *dsts: Callable[..., None], **kwargs) -> None:
        if False:
            return 10
        "\n        Indicates the next step to execute after this step has completed.\n\n        This statement should appear as the last statement of each step, except\n        the end step.\n\n        There are several valid formats to specify the next step:\n\n        - Straight-line connection: `self.next(self.next_step)` where `next_step` is a method in\n          the current class decorated with the `@step` decorator.\n\n        - Static fan-out connection: `self.next(self.step1, self.step2, ...)` where `stepX` are\n          methods in the current class decorated with the `@step` decorator.\n\n        - Foreach branch:\n          ```\n          self.next(self.foreach_step, foreach='foreach_iterator')\n          ```\n          In this situation, `foreach_step` is a method in the current class decorated with the\n          `@step` decorator and `foreach_iterator` is a variable name in the current class that\n          evaluates to an iterator. A task will be launched for each value in the iterator and\n          each task will execute the code specified by the step `foreach_step`.\n\n        Parameters\n        ----------\n        dsts : Method\n            One or more methods annotated with `@step`.\n\n        Raises\n        ------\n        InvalidNextException\n            Raised if the format of the arguments does not match one of the ones given above.\n        "
        step = self._current_step
        foreach = kwargs.pop('foreach', None)
        num_parallel = kwargs.pop('num_parallel', None)
        if kwargs:
            kw = next(iter(kwargs))
            msg = "Step *{step}* passes an unknown keyword argument '{invalid}' to self.next().".format(step=step, invalid=kw)
            raise InvalidNextException(msg)
        if self._transition is not None:
            msg = 'Multiple self.next() calls detected in step *{step}*. Call self.next() only once.'.format(step=step)
            raise InvalidNextException(msg)
        funcs = []
        for (i, dst) in enumerate(dsts):
            try:
                name = dst.__func__.__name__
            except:
                msg = 'In step *{step}* the {arg}. argument in self.next() is not a function. Make sure all arguments in self.next() are methods of the Flow class.'.format(step=step, arg=i + 1)
                raise InvalidNextException(msg)
            if not hasattr(self, name):
                msg = 'Step *{step}* specifies a self.next() transition to an unknown step, *{name}*.'.format(step=step, name=name)
                raise InvalidNextException(msg)
            funcs.append(name)
        if num_parallel is not None and num_parallel >= 1:
            if len(dsts) > 1:
                raise InvalidNextException('Only one destination allowed when num_parallel used in self.next()')
            foreach = '_parallel_ubf_iter'
            self._parallel_ubf_iter = ParallelUBF(num_parallel)
        if foreach:
            if not isinstance(foreach, basestring):
                msg = "Step *{step}* has an invalid self.next() transition. The argument to 'foreach' must be a string.".format(step=step)
                raise InvalidNextException(msg)
            if len(dsts) != 1:
                msg = "Step *{step}* has an invalid self.next() transition. Specify exactly one target for 'foreach'.".format(step=step)
                raise InvalidNextException(msg)
            try:
                foreach_iter = getattr(self, foreach)
            except:
                msg = 'Foreach variable *self.{var}* in step *{step}* does not exist. Check your variable.'.format(step=step, var=foreach)
                raise InvalidNextException(msg)
            if issubclass(type(foreach_iter), UnboundedForeachInput):
                self._unbounded_foreach = True
                self._foreach_num_splits = None
                self._validate_ubf_step(funcs[0])
            else:
                try:
                    self._foreach_num_splits = sum((1 for _ in foreach_iter))
                except TypeError:
                    msg = 'Foreach variable *self.{var}* in step *{step}* is not iterable. Check your variable.'.format(step=step, var=foreach)
                    raise InvalidNextException(msg)
                if self._foreach_num_splits == 0:
                    msg = 'Foreach iterator over *{var}* in step *{step}* produced zero splits. Check your variable.'.format(step=step, var=foreach)
                    raise InvalidNextException(msg)
            self._foreach_var = foreach
        if foreach is None:
            if len(dsts) < 1:
                msg = 'Step *{step}* has an invalid self.next() transition. Specify at least one step function as an argument in self.next().'.format(step=step)
                raise InvalidNextException(msg)
        self._transition = (funcs, foreach)

    def __str__(self):
        if False:
            return 10
        step_name = getattr(self, '_current_step', None)
        if step_name:
            index = ','.join((str(idx) for (idx, _, _) in self.foreach_stack()))
            if index:
                inp = self.input
                if inp is None:
                    return '<flow %s step %s[%s]>' % (self.name, step_name, index)
                else:
                    inp = str(inp)
                    if len(inp) > 20:
                        inp = inp[:20] + '...'
                    return '<flow %s step %s[%s] (input: %s)>' % (self.name, step_name, index, inp)
            else:
                return '<flow %s step %s>' % (self.name, step_name)
        else:
            return '<flow %s>' % self.name

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        raise MetaflowException("Flows can't be serialized. Maybe you tried to assign *self* or one of the *inputs* to an attribute? Instead of serializing the whole flow, you should choose specific attributes, e.g. *input.some_var*, to be stored.")