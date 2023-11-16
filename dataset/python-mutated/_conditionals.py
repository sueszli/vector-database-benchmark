"""This module contains the implementation of DALI if statement.

It initializes AutoGraph with the DaliOperatorOverload that provides the overload for the if_stmt
and adjust the filtered modules so DALI code is not converted.

The if_stmt provides access to both branches as callables and the set_state/get_state functions
that allows to capture and adjust all symbols modified within those branches. This allows to
checkpoint the state and visit the code of both branches.

if_stmt highlights which state variables are considered the outputs of the if/else pair - we can
use the state captured after visiting if and else branches and produce fn._conditional.merge
nodes for all of them.

When visiting the if/else scopes, we are tracking tha path that we took and the predicates that
were used via the _ConditionStack. As it is not easy to detect which state variables would be
consumed as inputs to DALI operators, we inject additional code to the operator function.
Every time a DataNode is consumed, we look up in which scope it was produced and travel the
path from that point to the current scope in the _ConditionStack, applying necessary splits.
All the return values are registered to the current scope for further lookups.
"""
from nvidia.dali import _autograph
from nvidia.dali.data_node import DataNode as _DataNode
from nvidia.dali import fn
from nvidia.dali._autograph.utils import ag_logging as logging
from nvidia.dali._autograph.operators import variables
from contextlib import contextmanager
from enum import Enum
import tree

def _data_node_repr(data_node):
    if False:
        return 10
    return f'DataNode(name={data_node.name}, device={data_node.device}, source={data_node.source})'

class _Branch(Enum):
    TrueBranch = 0
    FalseBranch = 1
    Undefined = 2

class _StackEntry:
    """Information about 1 nesting level of if/else statement.

    Keeps the current branch (if we entered if/else branch) and the data nodes that were
    produced in their scopes. Keeps the mapping of DataNodes produced in higher scopes that
    were already split for use in this scope.
    """

    def __init__(self, predicate):
        if False:
            while True:
                i = 10
        self.predicate = predicate
        self.branch = _Branch.Undefined
        self.splits = {}
        self.produced_true = set()
        self.produced_false = set()
        self.produced_special = set()

    @property
    def produced(self):
        if False:
            i = 10
            return i + 15
        'Access the set of hashes of DataNodes produced in the scope of currently selected branch.\n        '
        if self.branch == _Branch.TrueBranch:
            return self.produced_true
        elif self.branch == _Branch.FalseBranch:
            return self.produced_false
        else:
            return self.produced_special | self.produced_true | self.produced_false

    @produced.setter
    def produced(self, value):
        if False:
            print('Hello World!')
        'Access the set of hashes of DataNodes produced in the scope of currently selected branch.\n        '
        if self.branch == _Branch.TrueBranch:
            self.produced_true = value
        elif self.branch == _Branch.FalseBranch:
            self.produced_false = value
        else:
            self.produced_special = value

    def add_produced(self, data_node):
        if False:
            i = 10
            return i + 15
        'Add the DataNode or DataNodes to produced in the scope of currently selected branch.'
        if isinstance(data_node, _DataNode):
            self.produced |= {_data_node_repr(data_node)}
        elif isinstance(data_node, list):
            if not data_node:
                return
            if isinstance(data_node[0], _DataNode):
                self.produced |= set((_data_node_repr(dn) for dn in data_node))
            elif isinstance(data_node[0], list):
                flat_list = [item for sublist in data_node for item in sublist]
                self.add_produced(flat_list)
        else:
            raise ValueError(f'Unexpected operator result to register: {data_node}. Expected up to two-level nesting of DataNode.')

    def add_split(self, source_data_node, producer_node, true_node, false_node):
        if False:
            i = 10
            return i + 15
        'Register the outputs of split node that were produced from the source_data_node\n        (or its descendant on this scope, the shortcut node).\n\n        Parameters\n        ----------\n        source_data_node : DataNode\n            Original source node that was looked up, record for faster consecutive lookups\n        producer_node : DataNode\n            The closest node on the path from source_data_node to this split\n        true_node : DataNode\n            True branch split\n        false_node : DataNode\n            False branch split\n        '
        self.splits[_data_node_repr(source_data_node)] = (true_node, false_node)
        self.splits[_data_node_repr(producer_node)] = (true_node, false_node)
        self.produced_true |= {_data_node_repr(true_node)}
        self.produced_false |= {_data_node_repr(false_node)}

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return f'StackEntry: pred={self.predicate}, branch={self.branch}, splits={self.splits}, produced={self.produced}'

    def has(self, data_node):
        if False:
            while True:
                i = 10
        'Check if this DataNode was either produced in this scope or already split for this scope.\n        '
        if _data_node_repr(data_node) in self.produced:
            return True
        elif _data_node_repr(data_node) in self.splits:
            return True
        else:
            return False

    def get(self, data_node):
        if False:
            return 10
        'Return the `data_node` if it was produced in this scope, or the appropriate split node\n        that was created for accessing the `data_node` in this scope.\n        '
        assert self.has(data_node)
        if _data_node_repr(data_node) in self.produced:
            return data_node
        else:
            assert self.branch in {_Branch.TrueBranch, _Branch.FalseBranch}
            return self.splits[_data_node_repr(data_node)][self.branch.value]

class _ConditionStack:
    """Tracks the current if/else scope with the path that we took. Captures the used and produced
    data nodes, applying the necessary splits based on the scope level where they were produced
    and where they are used.
    """

    def __init__(self):
        if False:
            return 10
        self._stack = [_StackEntry(None)]
        self._is_registration_allowed = True

    def push_predicate(self, predicate):
        if False:
            print('Hello World!')
        'Add next level of if/else scope that is predicated with the `predicate`.\n        The user might have provided a predicate from a scope of higher level, which means\n        that `predicate` might be subject to additional slicing. Apply that slicing and return\n        the actual predicate that will be used for slicing when entering this scope.\n\n        The situation will happen for example in a case like this, where both predicates are\n        produced in global scope:\n\n        pred_0 = ...\n        pred_1 = ...\n\n        if pred_0:  # push_pred(pred_0) -> returns pred_0\n            if pred_1:  # push_pred(pred_1) ->\n                        # -> returns fn._conditional.slice(pred_1, predicate=pred_0)\n\n        Parameters\n        ----------\n        predicate : DataNode\n            Predicate guarding this scope.\n\n        Returns\n        -------\n        DataNode\n            Actual predicate after applying necessary slices to use it in this scope.\n        '
        new_pred = self.preprocess_input(predicate)
        new_entry = _StackEntry(new_pred)
        self._stack.append(new_entry)
        return new_pred

    def top(self):
        if False:
            return 10
        'Get the top scope in the stack'
        return self._stack[-1]

    def pop(self):
        if False:
            return 10
        'Remove the top scope from the stack'
        result = self._stack.pop()
        return result

    def stack_depth(self):
        if False:
            return 10
        'Get the depth of the stack. Note, that by default there is at least one element\n        - the global scope.'
        return len(self._stack)

    def _find_closest(self, data_node):
        if False:
            print('Hello World!')
        'Find the closest scope level in the stack where we can access this node as produced\n        (or the split of this node closest to us).\n        '
        for level in range(self.stack_depth() - 1, -1, -1):
            if self._stack[level].has(data_node):
                return level
        raise ValueError(f'{data_node} was not produced within this trace.')

    def _realize_split(self, data_node, stack_level):
        if False:
            i = 10
            return i + 15
        'The data_node was produced (or last accessed as via split) in scope earlier than the\n        current one, traverse the scopes between that level and current one, and insert split nodes.\n\n        Parameters\n        ----------\n        data_node : DataNode\n            The data node that we want to use in the current scope.\n        stack_level : int\n            Stack level where the data_node was last "seen".\n\n        Returns\n        -------\n        DataNode\n            New node that can be used in current branch and scope.\n        '
        assert 0 <= stack_level and stack_level < self.stack_depth() - 1
        produced_data_node = self._stack[stack_level].get(data_node)
        bottom = self._stack[:stack_level + 1]
        top = self._stack[stack_level + 1:]
        self._stack = bottom
        while top:
            current_entry = top.pop(0)
            predicate = current_entry.predicate
            logging.log(9, f'{self._indent()}[IF] Inserting split at {self.stack_depth() - 1}: split({produced_data_node}, predicate={predicate}.')
            self._is_registration_allowed = False
            (true_node, false_node) = fn._conditional.split(produced_data_node, predicate=predicate, _if_stmt=True)
            self._is_registration_allowed = True
            current_entry.add_split(data_node, produced_data_node, true_node, false_node)
            if current_entry.branch == _Branch.TrueBranch:
                produced_data_node = true_node
            else:
                produced_data_node = false_node
            self._stack.append(current_entry)
        return produced_data_node

    def preprocess_input(self, data_node):
        if False:
            print('Hello World!')
        'Process the DataNode that is an input to an operator call. Detect if the DataNode was\n        produced on the same nesting level. If not, split accordingly to the stack of the previous\n        conditions. Caches the previously processed DataNodes to not do repeated splitting.\n        '
        stack_level = self._find_closest(data_node)
        logging.log(8, f'{self._indent()}[IF/Input] {data_node} accessed at level {self.stack_depth() - 1} found at {stack_level}.')
        if stack_level == self.stack_depth() - 1:
            return self.top().get(data_node)
        return self._realize_split(data_node, stack_level)

    def register_data_nodes(self, data_nodes, global_scope=False):
        if False:
            for i in range(10):
                print('nop')
        'Register the data nodes as produced in current scope, otherwise if `global_scope` is True\n        put them in the outermost scope.\n        '
        if not self._is_registration_allowed:
            return
        logging.log(8, f'{self._indent()}[IF/Register] {data_nodes} at {self.stack_depth() - 1}')
        scope = self._stack[0] if global_scope else self.top()
        tree.map_structure(lambda node: scope.add_produced(node), data_nodes)

    def track_true_branch(self):
        if False:
            print('Hello World!')
        'Mark `if` (true) branch as current scope.'
        self.top().branch = _Branch.TrueBranch

    def track_false_branch(self):
        if False:
            print('Hello World!')
        'Mark `else` (false) branch as current scope.'
        self.top().branch = _Branch.FalseBranch

    def no_branch(self):
        if False:
            return 10
        'Mark no branch being tracked, the scope "level" stays related to the same if/else\n        statement.'
        self.top().branch = _Branch.Undefined

    def track_merge(self, split_predicate):
        if False:
            i = 10
            return i + 15
        "Enter the merge section of the if/else statement. It adds the corresponding\n        split_predicate to the nodes visible as produced in the current scope, so all data nodes\n        are directly accessible in this scope when looked up by the merge operator.\n        We don't care about removing it as it's the last thing happening in that statement.\n        "
        self.no_branch()
        self.top().add_produced(split_predicate)

    def scope_batch_size_tracker(self):
        if False:
            i = 10
            return i + 15
        'Return the DataNode that can be used as a reference batch size in this scope.\n        None is returned if we are in the top level scope.\n        '
        if self.stack_depth() == 1:
            return None
        if self.top().branch in {_Branch.TrueBranch, _Branch.FalseBranch}:
            return self.preprocess_input(self.top().predicate)
        else:
            return self.top().predicate

    def _indent(self):
        if False:
            while True:
                i = 10
        'Helper for indenting the log messages to resemble visited scopes'
        return '  ' * (self.stack_depth() - 1)

@contextmanager
def _cond_manager(predicate):
    if False:
        i = 10
        return i + 15
    actual_predicate = this_condition_stack().push_predicate(predicate)
    logging.log(7, f'{this_condition_stack()._indent()}[IF]: {predicate} at {this_condition_stack().stack_depth() - 1}')
    yield actual_predicate
    this_condition_stack().pop()

@contextmanager
def _cond_true():
    if False:
        for i in range(10):
            print('nop')
    this_condition_stack().track_true_branch()
    logging.log(7, f'{this_condition_stack()._indent()}[IF]: `if` branch at {this_condition_stack().stack_depth() - 1}')
    yield
    this_condition_stack().no_branch()

@contextmanager
def _cond_false():
    if False:
        i = 10
        return i + 15
    this_condition_stack().track_false_branch()
    logging.log(7, f'{this_condition_stack()._indent()}[IF]: `else` branch at {this_condition_stack().stack_depth() - 1}')
    yield
    this_condition_stack().no_branch()

@contextmanager
def _cond_merge(split_predicate):
    if False:
        print('Hello World!')
    this_condition_stack().track_merge(split_predicate)
    yield
    this_condition_stack().no_branch()

def conditionals_enabled():
    if False:
        while True:
            i = 10
    'Check (within a Pipeline context) if the conditionals are enabled.\n    '
    from nvidia.dali._debug_mode import _PipelineDebug
    current_pipeline = _PipelineDebug.current()
    enabled = getattr(current_pipeline, '_conditionals_enabled', False)
    return enabled

def this_condition_stack():
    if False:
        for i in range(10):
            print('nop')
    'Return the condition stack of current Pipeline'
    from nvidia.dali._debug_mode import _PipelineDebug
    current_pipeline = _PipelineDebug.current()
    if current_pipeline._condition_stack is None:
        raise ValueError('Cannot access current condition stack when conditionals were not enabled for a given pipeline.')
    return current_pipeline._condition_stack

def register_data_nodes(data_node, inputs=[], args={}):
    if False:
        for i in range(10):
            print('nop')
    'Register the outputs of the operator as produced in the scope of the current conditional\n    branch. Pass the list of inputs and dictionary of arguments to automatically detect if any\n    DataNode was passed to that operator, indicating that it has proper inputs or argument inputs\n    and can infer the batch size. Otherwise the outputs are registered in global scope, assuming\n    that they use current batch size.\n\n    Parameters\n    ----------\n    data_node : DataNode or a list/tuple of DataNode\n        The output of the operator to be registered.\n    inputs : List of DataNode\n        Optional list of inputs of the operator whose outputs we are registering.\n    args : Dict of DataNode\n        Optional dictionary containing the arguments of the operator whose outputs we are\n        registering.\n    '
    any_positional_input = any((isinstance(input, _DataNode) for input in inputs))
    any_arg_input = any((isinstance(arg, _DataNode) for (arg_name, arg) in args.items()))
    any_input = any_positional_input or any_arg_input
    this_condition_stack().register_data_nodes(data_node, global_scope=not any_input)

def inject_implicit_scope_argument(schema, kwargs):
    if False:
        return 10
    '\n    Adds hidden _scope argument to the inputless operators whose outputs for\n    any given sample depend on the actual batch size, e.g. fn.batch_permutation.\n    '
    if schema.HasArgument('_scope'):
        conditional_scope = this_condition_stack()
        scope_masked_batch = conditional_scope.scope_batch_size_tracker()
        kwargs['_scope'] = scope_masked_batch

def apply_conditional_split(input):
    if False:
        for i in range(10):
            print('nop')
    'Preprocess the DataNode to obtain correctly split batch for the current if scope.'
    return this_condition_stack().preprocess_input(input)

def apply_conditional_split_to_branch_outputs(branch_outputs, promote_constants=True):
    if False:
        i = 10
        return i + 15
    'Apply splitting to the branch outputs. This may be necessary for DataNodes that are\n    branch outputs but were not touched in that branch (for example that branch is no-op).\n\n    Parameters\n    ----------\n    branch_outputs : tuple of DataNode\n        Outputs of the branch\n    promote_constants : bool, optional\n        Whether to promote constants to cpu-based Constant op, by default True\n\n    Returns\n    -------\n    tuple of DataNode\n    '
    from nvidia.dali.types import Constant

    def apply_split(atom):
        if False:
            return 10
        if isinstance(atom, _DataNode):
            return apply_conditional_split(atom)
        elif promote_constants:
            constant_node = Constant(atom, device='cpu')
            register_data_nodes(constant_node)
            return apply_conditional_split(constant_node)
        return atom
    return tree.map_structure(apply_split, branch_outputs)

def apply_conditional_split_to_args(inputs, kwargs):
    if False:
        while True:
            i = 10
    'Preprocess the inputs and kwargs of the operator to obtain correctly split inputs for the\n    current if scope.'
    inputs = apply_conditional_split_to_branch_outputs(inputs, False)
    for (key, arg) in kwargs.items():
        if isinstance(arg, _DataNode):
            kwargs[key] = apply_conditional_split(arg)
    return (inputs, kwargs)

def _verify_branch_outputs(outputs, symbol_names, branch_name):
    if False:
        for i in range(10):
            print('nop')
    'Verifies variables output by a conditional branch for consistency.'
    common_explanation = 'Encountered inconsistent outputs out of the `if/else` control flow statement. Variables need to be initialized in every code path (both `if` branches).'
    for (name, output) in zip(symbol_names, outputs):
        if isinstance(output, variables.Undefined):
            raise RuntimeError(f"{common_explanation} Variable '{name}' must also be initialized in the `{branch_name}` branch.")
        if isinstance(output, variables.UndefinedReturnValue):
            raise RuntimeError(f'{common_explanation} The `{branch_name}` branch must also have a return statement.')

class DaliOperatorOverload(_autograph.OperatorBase):

    def detect_overload_ld(self, v):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(v, _DataNode)

    def ld(self, v):
        if False:
            return 10
        branch_v = apply_conditional_split(v)
        return branch_v

    def detect_overload_if_stmt(self, cond):
        if False:
            print('Hello World!')
        return isinstance(cond, _DataNode)

    def if_stmt(self, cond, body, orelse, get_state, set_state, symbol_names, nouts):
        if False:
            return 10
        init_state = get_state()
        with _cond_manager(cond) as split_predicate:
            with _cond_true():
                body()
                body_state = get_state()
                _verify_branch_outputs(body_state, symbol_names, 'if')
                body_outputs = body_state[:nouts]
                body_outputs = apply_conditional_split_to_branch_outputs(body_outputs)
            set_state(init_state)
            with _cond_false():
                orelse()
                orelse_state = get_state()
                _verify_branch_outputs(orelse_state, symbol_names, 'else')
                orelse_outputs = orelse_state[:nouts]
                orelse_outputs = apply_conditional_split_to_branch_outputs(orelse_outputs)
            output_values = []
            with _cond_merge(split_predicate):
                err_msg = 'Divergent data found in different branches of `if/else` control flow statement. Variables in all code paths are merged into common output batches. The values assigned to a given variable need to have the same nesting structure in every code path (both `if` branches).\nFor example, if we define a variable as a tuple in one branch, it must be defined as a tuple of the same length in the other branch - the contents of the tuples may be different. If we define a variable as a dictionary, the other branch must define it as a dictionary with the same set of keys, the values may be different.\n'
                try:
                    tree.assert_same_structure(body_outputs, orelse_outputs, check_types=True)
                except ValueError as e:
                    raise ValueError(err_msg + str(e)) from None
                except TypeError as e:
                    raise TypeError(err_msg + str(e)) from None

                def merge_branches(new_body_val, new_orelse_val):
                    if False:
                        while True:
                            i = 10
                    logging.log(9, f'{this_condition_stack()._indent()}[IF] Inserting merge at {this_condition_stack().stack_depth() - 1}: merge({new_body_val}, {new_orelse_val}, predicate={split_predicate}.')
                    return fn._conditional.merge(new_body_val, new_orelse_val, predicate=split_predicate)
                output_values = tree.map_structure(merge_branches, body_outputs, orelse_outputs)
        this_condition_stack().register_data_nodes(output_values, False)
        output_values += init_state[nouts:]
        set_state(output_values)

    def detect_overload_not_(self, a):
        if False:
            i = 10
            return i + 15
        return isinstance(a, _DataNode)

    def not_(self, a):
        if False:
            i = 10
            return i + 15
        return fn._conditional.not_(a)

    def detect_overload_lazy_and(self, a):
        if False:
            return 10
        return isinstance(a, _DataNode)

    def lazy_and(self, a_value, b):
        if False:
            i = 10
            return i + 15
        a_validated = fn._conditional.validate_logical(a_value, expression_name='and', expression_side='left')
        with _cond_manager(a_validated) as split_predicate:
            with _cond_true():
                b_value = b()
                b_validated = fn._conditional.validate_logical(b_value, expression_name='and', expression_side='right')
                body_outputs = apply_conditional_split(b_validated)
            with _cond_false():
                else_outputs = apply_conditional_split(split_predicate)
            with _cond_merge(split_predicate):
                merged = fn._conditional.merge(body_outputs, else_outputs, predicate=split_predicate)
        this_condition_stack().register_data_nodes([merged], False)
        return merged

    def detect_overload_lazy_or(self, a):
        if False:
            while True:
                i = 10
        return isinstance(a, _DataNode)

    def lazy_or(self, a_value, b):
        if False:
            return 10
        a_validated = fn._conditional.validate_logical(a_value, expression_name='or', expression_side='left')
        with _cond_manager(a_validated) as split_predicate:
            with _cond_true():
                body_outputs = apply_conditional_split(split_predicate)
            with _cond_false():
                b_value = b()
                b_validated = fn._conditional.validate_logical(b_value, expression_name='or', expression_side='right')
                else_outputs = apply_conditional_split(b_validated)
            with _cond_merge(split_predicate):
                merged = fn._conditional.merge(body_outputs, else_outputs, predicate=split_predicate)
        this_condition_stack().register_data_nodes([merged], False)
        return merged
_OVERLOADS = DaliOperatorOverload()
_autograph.initialize_autograph(_OVERLOADS, convert_modules=['nvidia.dali.auto_aug'], do_not_convert_modules=['nvidia.dali._autograph', 'nvidia.dali'])