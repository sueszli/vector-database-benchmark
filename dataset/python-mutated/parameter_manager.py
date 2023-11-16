""""Management of pulse program parameters.

Background
==========

In contrast to ``QuantumCircuit``, in pulse programs, parameter objects can be stored in
multiple places at different layers, for example

- program variables: ``ScheduleBlock.alignment_context._context_params``

- instruction operands: ``ShiftPhase.phase``, ...

- operand parameters: ``pulse.parameters``, ``channel.index`` ...

This complexity is due to the tight coupling of the program to an underlying device Hamiltonian,
i.e. the variance of physical parameters between qubits and their couplings.
If we want to define a program that can be used with arbitrary qubits,
we should be able to parametrize every control parameter in the program.

Implementation
==============

Managing parameters in each object within a program, i.e. the ``ParameterTable`` model,
makes the framework quite complicated. With the ``ParameterManager`` class within this module,
the parameter assignment operation is performed by a visitor instance.

The visitor pattern is a way of separating data processing from the object on which it operates.
This removes the overhead of parameter management from each piece of the program.
The computational complexity of the parameter assignment operation may be increased
from the parameter table model of ~O(1), however, usually, this calculation occurs
only once before the program is executed. Thus this doesn't hurt user experience during
pulse programming. On the contrary, it removes parameter table object and associated logic
from each object, yielding smaller object creation cost and higher performance
as the data amount scales.

Note that we don't need to write any parameter management logic for each object,
and thus this parameter framework gives greater scalability to the pulse module.
"""
from copy import copy
from typing import List, Dict, Set, Any, Union
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse import instructions, channels
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.library import ParametricPulse, SymbolicPulse, Waveform
from qiskit.pulse.schedule import Schedule, ScheduleBlock
from qiskit.pulse.transforms.alignments import AlignmentKind
from qiskit.pulse.utils import format_parameter_value

class NodeVisitor:
    """A node visitor base class that walks instruction data in a pulse program and calls
    visitor functions for every node.

    Though this class implementation is based on Python AST, each node doesn't have
    a dedicated node class due to the lack of an abstract syntax tree for pulse programs in
    Qiskit. Instead of parsing pulse programs, this visitor class finds the associated visitor
    function based on class name of the instruction node, i.e. ``Play``, ``Call``, etc...
    The `.visit` method recursively checks superclass of given node since some parametrized
    components such as ``DriveChannel`` may share a common superclass with other subclasses.
    In this example, we can just define ``visit_Channel`` method instead of defining
    the same visitor function for every subclasses.

    Some instructions may have special logic or data structure to store parameter objects,
    and visitor functions for these nodes should be individually defined.

    Because pulse programs can be nested into another pulse program,
    the visitor function should be able to recursively call proper visitor functions.
    If visitor function is not defined for a given node, ``generic_visit``
    method is called. Usually, this method is provided for operating on object defined
    outside of the Qiskit Pulse module.
    """

    def visit(self, node: Any):
        if False:
            return 10
        'Visit a node.'
        visitor = self._get_visitor(type(node))
        return visitor(node)

    def _get_visitor(self, node_class):
        if False:
            while True:
                i = 10
        'A helper function to recursively investigate superclass visitor method.'
        if node_class == object:
            return self.generic_visit
        try:
            return getattr(self, f'visit_{node_class.__name__}')
        except AttributeError:
            return self._get_visitor(node_class.__base__)

    def visit_ScheduleBlock(self, node: ScheduleBlock):
        if False:
            i = 10
            return i + 15
        'Visit ``ScheduleBlock``. Recursively visit context blocks and overwrite.\n\n        .. note:: ``ScheduleBlock`` can have parameters in blocks and its alignment.\n        '
        raise NotImplementedError

    def visit_Schedule(self, node: Schedule):
        if False:
            for i in range(10):
                print('nop')
        'Visit ``Schedule``. Recursively visit schedule children and overwrite.'
        raise NotImplementedError

    def generic_visit(self, node: Any):
        if False:
            i = 10
            return i + 15
        'Called if no explicit visitor function exists for a node.'
        raise NotImplementedError

class ParameterSetter(NodeVisitor):
    """Node visitor for parameter binding.

    This visitor is initialized with a dictionary of parameters to be assigned,
    and assign values to operands of nodes found.
    """

    def __init__(self, param_map: Dict[ParameterExpression, ParameterValueType]):
        if False:
            while True:
                i = 10
        self._param_map = param_map

    def visit_ScheduleBlock(self, node: ScheduleBlock):
        if False:
            for i in range(10):
                print('nop')
        'Visit ``ScheduleBlock``. Recursively visit context blocks and overwrite.\n\n        .. note:: ``ScheduleBlock`` can have parameters in blocks and its alignment.\n        '
        node._alignment_context = self.visit_AlignmentKind(node.alignment_context)
        for elm in node._blocks:
            self.visit(elm)
        self._update_parameter_manager(node)
        return node

    def visit_Schedule(self, node: Schedule):
        if False:
            while True:
                i = 10
        'Visit ``Schedule``. Recursively visit schedule children and overwrite.'
        node._Schedule__children = [(t0, self.visit(sched)) for (t0, sched) in node.instructions]
        node._renew_timeslots()
        self._update_parameter_manager(node)
        return node

    def visit_AlignmentKind(self, node: AlignmentKind):
        if False:
            print('Hello World!')
        "Assign parameters to block's ``AlignmentKind`` specification."
        new_parameters = tuple((self.visit(param) for param in node._context_params))
        node._context_params = new_parameters
        return node

    def visit_Call(self, node: instructions.Call):
        if False:
            for i in range(10):
                print('nop')
        "Assign parameters to ``Call`` instruction.\n\n        .. note:: ``Call`` instruction has a special parameter handling logic.\n            This instruction separately keeps program, i.e. parametrized schedule,\n            and bound parameters until execution. The parameter assignment operation doesn't\n            immediately override its operand data.\n        "
        if node.is_parameterized():
            new_table = copy(node.arguments)
            for (parameter, value) in new_table.items():
                if isinstance(value, ParameterExpression):
                    new_table[parameter] = self._assign_parameter_expression(value)
            node.arguments = new_table
        return node

    def visit_Instruction(self, node: instructions.Instruction):
        if False:
            i = 10
            return i + 15
        'Assign parameters to general pulse instruction.\n\n        .. note:: All parametrized object should be stored in the operands.\n            Otherwise parameter cannot be detected.\n        '
        if node.is_parameterized():
            node._operands = tuple((self.visit(op) for op in node.operands))
        return node

    def visit_Channel(self, node: channels.Channel):
        if False:
            print('Hello World!')
        'Assign parameters to ``Channel`` object.'
        if node.is_parameterized():
            new_index = self._assign_parameter_expression(node.index)
            if not isinstance(new_index, ParameterExpression):
                if not isinstance(new_index, int) or new_index < 0:
                    raise PulseError('Channel index must be a nonnegative integer')
            return node.__class__(index=new_index)
        return node

    def visit_ParametricPulse(self, node: ParametricPulse):
        if False:
            while True:
                i = 10
        'Assign parameters to ``ParametricPulse`` object.'
        if node.is_parameterized():
            new_parameters = {}
            for (op, op_value) in node.parameters.items():
                if isinstance(op_value, ParameterExpression):
                    op_value = self._assign_parameter_expression(op_value)
                new_parameters[op] = op_value
            return node.__class__(**new_parameters, name=node.name)
        return node

    def visit_SymbolicPulse(self, node: SymbolicPulse):
        if False:
            print('Hello World!')
        'Assign parameters to ``SymbolicPulse`` object.'
        if node.is_parameterized():
            if isinstance(node.duration, ParameterExpression):
                node.duration = self._assign_parameter_expression(node.duration)
            for name in node._params:
                pval = node._params[name]
                if isinstance(pval, ParameterExpression):
                    new_val = self._assign_parameter_expression(pval)
                    node._params[name] = new_val
            node.validate_parameters()
        return node

    def visit_Waveform(self, node: Waveform):
        if False:
            while True:
                i = 10
        'Assign parameters to ``Waveform`` object.\n\n        .. node:: No parameter can be assigned to ``Waveform`` object.\n        '
        return node

    def generic_visit(self, node: Any):
        if False:
            return 10
        "Assign parameters to object that doesn't belong to Qiskit Pulse module."
        if isinstance(node, ParameterExpression):
            return self._assign_parameter_expression(node)
        else:
            return node

    def _assign_parameter_expression(self, param_expr: ParameterExpression):
        if False:
            return 10
        'A helper function to assign parameter value to parameter expression.'
        new_value = copy(param_expr)
        updated = param_expr.parameters & self._param_map.keys()
        for param in updated:
            new_value = new_value.assign(param, self._param_map[param])
        new_value = format_parameter_value(new_value)
        return new_value

    def _update_parameter_manager(self, node: Union[Schedule, ScheduleBlock]):
        if False:
            return 10
        'A helper function to update parameter manager of pulse program.'
        if not hasattr(node, '_parameter_manager'):
            raise PulseError(f'Node type {node.__class__.__name__} has no parameter manager.')
        param_manager = node._parameter_manager
        updated = param_manager.parameters & self._param_map.keys()
        new_parameters = set()
        for param in param_manager.parameters:
            if param not in updated:
                new_parameters.add(param)
                continue
            new_value = self._param_map[param]
            if isinstance(new_value, ParameterExpression):
                new_parameters |= new_value.parameters
        param_manager._parameters = new_parameters

class ParameterGetter(NodeVisitor):
    """Node visitor for parameter finding.

    This visitor initializes empty parameter array, and recursively visits nodes
    and add parameters found to the array.
    """

    def __init__(self):
        if False:
            return 10
        self.parameters = set()

    def visit_ScheduleBlock(self, node: ScheduleBlock):
        if False:
            print('Hello World!')
        'Visit ``ScheduleBlock``. Recursively visit context blocks and search parameters.\n\n        .. note:: ``ScheduleBlock`` can have parameters in blocks and its alignment.\n        '
        self.parameters |= node._parameter_manager.parameters

    def visit_Schedule(self, node: Schedule):
        if False:
            for i in range(10):
                print('nop')
        'Visit ``Schedule``. Recursively visit schedule children and search parameters.'
        self.parameters |= node.parameters

    def visit_AlignmentKind(self, node: AlignmentKind):
        if False:
            for i in range(10):
                print('nop')
        "Get parameters from block's ``AlignmentKind`` specification."
        for param in node._context_params:
            if isinstance(param, ParameterExpression):
                self.parameters |= param.parameters

    def visit_Call(self, node: instructions.Call):
        if False:
            i = 10
            return i + 15
        'Get parameters from ``Call`` instruction.\n\n        .. note:: ``Call`` instruction has a special parameter handling logic.\n            This instruction separately keeps parameters and program.\n        '
        self.parameters |= node.parameters

    def visit_Instruction(self, node: instructions.Instruction):
        if False:
            for i in range(10):
                print('nop')
        'Get parameters from general pulse instruction.\n\n        .. note:: All parametrized object should be stored in the operands.\n            Otherwise, parameter cannot be detected.\n        '
        for op in node.operands:
            self.visit(op)

    def visit_Channel(self, node: channels.Channel):
        if False:
            i = 10
            return i + 15
        'Get parameters from ``Channel`` object.'
        self.parameters |= node.parameters

    def visit_ParametricPulse(self, node: ParametricPulse):
        if False:
            print('Hello World!')
        'Get parameters from ``ParametricPulse`` object.'
        for op_value in node.parameters.values():
            if isinstance(op_value, ParameterExpression):
                self.parameters |= op_value.parameters

    def visit_SymbolicPulse(self, node: SymbolicPulse):
        if False:
            i = 10
            return i + 15
        'Get parameters from ``SymbolicPulse`` object.'
        for op_value in node.parameters.values():
            if isinstance(op_value, ParameterExpression):
                self.parameters |= op_value.parameters

    def visit_Waveform(self, node: Waveform):
        if False:
            return 10
        'Get parameters from ``Waveform`` object.\n\n        .. node:: No parameter can be assigned to ``Waveform`` object.\n        '
        pass

    def generic_visit(self, node: Any):
        if False:
            while True:
                i = 10
        "Get parameters from object that doesn't belong to Qiskit Pulse module."
        if isinstance(node, ParameterExpression):
            self.parameters |= node.parameters

class ParameterManager:
    """Helper class to manage parameter objects associated with arbitrary pulse programs.

    This object is implicitly initialized with the parameter object storage
    that stores parameter objects added to the parent pulse program.

    Parameter assignment logic is implemented based on the visitor pattern.
    Instruction data and its location are not directly associated with this object.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        'Create new parameter table for pulse programs.'
        self._parameters = set()

    @property
    def parameters(self) -> Set[Parameter]:
        if False:
            while True:
                i = 10
        'Parameters which determine the schedule behavior.'
        return self._parameters

    def clear(self):
        if False:
            print('Hello World!')
        'Remove the parameters linked to this manager.'
        self._parameters.clear()

    def is_parameterized(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Return True iff the instruction is parameterized.'
        return bool(self.parameters)

    def get_parameters(self, parameter_name: str) -> List[Parameter]:
        if False:
            i = 10
            return i + 15
        'Get parameter object bound to this schedule by string name.\n\n        Because different ``Parameter`` objects can have the same name,\n        this method returns a list of ``Parameter`` s for the provided name.\n\n        Args:\n            parameter_name: Name of parameter.\n\n        Returns:\n            Parameter objects that have corresponding name.\n        '
        return [param for param in self.parameters if param.name == parameter_name]

    def assign_parameters(self, pulse_program: Any, value_dict: Dict[ParameterExpression, ParameterValueType]) -> Any:
        if False:
            return 10
        'Modify and return program data with parameters assigned according to the input.\n\n        Args:\n            pulse_program: Arbitrary pulse program associated with this manager instance.\n            value_dict: A mapping from Parameters to either numeric values or another\n                Parameter expression.\n\n        Returns:\n            Updated program data.\n        '
        valid_map = {k: value_dict[k] for k in value_dict.keys() & self._parameters}
        if valid_map:
            visitor = ParameterSetter(param_map=valid_map)
            return visitor.visit(pulse_program)
        return pulse_program

    def update_parameter_table(self, new_node: Any):
        if False:
            while True:
                i = 10
        'A helper function to update parameter table with given data node.\n\n        Args:\n            new_node: A new data node to be added.\n        '
        visitor = ParameterGetter()
        visitor.visit(new_node)
        self._parameters |= visitor.parameters