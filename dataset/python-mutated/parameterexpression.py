"""
ParameterExpression Class to enable creating simple expressions of Parameters.
"""
from __future__ import annotations
from typing import Callable, Union
import numbers
import operator
import numpy
from qiskit.circuit.exceptions import CircuitError
from qiskit.utils import optionals as _optionals
ParameterValueType = Union['ParameterExpression', float]

class ParameterExpression:
    """ParameterExpression class to enable creating expressions of Parameters."""
    __slots__ = ['_parameter_symbols', '_parameter_keys', '_symbol_expr', '_name_map']

    def __init__(self, symbol_map: dict, expr):
        if False:
            i = 10
            return i + 15
        'Create a new :class:`ParameterExpression`.\n\n        Not intended to be called directly, but to be instantiated via operations\n        on other :class:`Parameter` or :class:`ParameterExpression` objects.\n\n        Args:\n            symbol_map (Dict[Parameter, [ParameterExpression, float, or int]]):\n                Mapping of :class:`Parameter` instances to the :class:`sympy.Symbol`\n                serving as their placeholder in expr.\n            expr (sympy.Expr): Expression of :class:`sympy.Symbol` s.\n        '
        self._parameter_symbols = symbol_map
        self._parameter_keys = frozenset((p._hash_key() for p in self._parameter_symbols))
        self._symbol_expr = expr
        self._name_map: dict | None = None

    @property
    def parameters(self) -> set:
        if False:
            i = 10
            return i + 15
        'Returns a set of the unbound Parameters in the expression.'
        return self._parameter_symbols.keys()

    @property
    def _names(self) -> dict:
        if False:
            print('Hello World!')
        'Returns a mapping of parameter names to Parameters in the expression.'
        if self._name_map is None:
            self._name_map = {p.name: p for p in self._parameter_symbols}
        return self._name_map

    def conjugate(self) -> 'ParameterExpression':
        if False:
            for i in range(10):
                print('nop')
        'Return the conjugate.'
        if _optionals.HAS_SYMENGINE:
            import symengine
            conjugated = ParameterExpression(self._parameter_symbols, symengine.conjugate(self._symbol_expr))
        else:
            conjugated = ParameterExpression(self._parameter_symbols, self._symbol_expr.conjugate())
        return conjugated

    def assign(self, parameter, value: ParameterValueType) -> 'ParameterExpression':
        if False:
            i = 10
            return i + 15
        '\n        Assign one parameter to a value, which can either be numeric or another parameter\n        expression.\n\n        Args:\n            parameter (Parameter): A parameter in this expression whose value will be updated.\n            value: The new value to bind to.\n\n        Returns:\n            A new expression parameterized by any parameters which were not bound by assignment.\n        '
        if isinstance(value, ParameterExpression):
            return self.subs({parameter: value})
        return self.bind({parameter: value})

    def bind(self, parameter_values: dict, allow_unknown_parameters: bool=False) -> 'ParameterExpression':
        if False:
            while True:
                i = 10
        'Binds the provided set of parameters to their corresponding values.\n\n        Args:\n            parameter_values: Mapping of Parameter instances to the numeric value to which\n                              they will be bound.\n            allow_unknown_parameters: If ``False``, raises an error if ``parameter_values``\n                contains Parameters in the keys outside those present in the expression.\n                If ``True``, any such parameters are simply ignored.\n\n        Raises:\n            CircuitError:\n                - If parameter_values contains Parameters outside those in self.\n                - If a non-numeric value is passed in parameter_values.\n            ZeroDivisionError:\n                - If binding the provided values requires division by zero.\n\n        Returns:\n            A new expression parameterized by any parameters which were not bound by\n            parameter_values.\n        '
        if not allow_unknown_parameters:
            self._raise_if_passed_unknown_parameters(parameter_values.keys())
        self._raise_if_passed_nan(parameter_values)
        symbol_values = {}
        for (parameter, value) in parameter_values.items():
            if (param_expr := self._parameter_symbols.get(parameter)) is not None:
                symbol_values[param_expr] = value
        bound_symbol_expr = self._symbol_expr.subs(symbol_values)
        free_parameters = self.parameters - parameter_values.keys()
        free_parameter_symbols = {p: s for (p, s) in self._parameter_symbols.items() if p in free_parameters}
        if hasattr(bound_symbol_expr, 'is_infinite') and bound_symbol_expr.is_infinite or bound_symbol_expr == float('inf'):
            raise ZeroDivisionError('Binding provided for expression results in division by zero (Expression: {}, Bindings: {}).'.format(self, parameter_values))
        return ParameterExpression(free_parameter_symbols, bound_symbol_expr)

    def subs(self, parameter_map: dict, allow_unknown_parameters: bool=False) -> 'ParameterExpression':
        if False:
            while True:
                i = 10
        'Returns a new Expression with replacement Parameters.\n\n        Args:\n            parameter_map: Mapping from Parameters in self to the ParameterExpression\n                           instances with which they should be replaced.\n            allow_unknown_parameters: If ``False``, raises an error if ``parameter_map``\n                contains Parameters in the keys outside those present in the expression.\n                If ``True``, any such parameters are simply ignored.\n\n        Raises:\n            CircuitError:\n                - If parameter_map contains Parameters outside those in self.\n                - If the replacement Parameters in parameter_map would result in\n                  a name conflict in the generated expression.\n\n        Returns:\n            A new expression with the specified parameters replaced.\n        '
        if not allow_unknown_parameters:
            self._raise_if_passed_unknown_parameters(parameter_map.keys())
        inbound_names = {p.name: p for replacement_expr in parameter_map.values() for p in replacement_expr.parameters}
        self._raise_if_parameter_names_conflict(inbound_names, parameter_map.keys())
        new_parameter_symbols = {p: s for (p, s) in self._parameter_symbols.items() if p not in parameter_map}
        if _optionals.HAS_SYMENGINE:
            import symengine
            symbol_type = symengine.Symbol
        else:
            from sympy import Symbol
            symbol_type = Symbol
        symbol_map = {}
        for (old_param, new_param) in parameter_map.items():
            if (old_symbol := self._parameter_symbols.get(old_param)) is not None:
                symbol_map[old_symbol] = new_param._symbol_expr
                for p in new_param.parameters:
                    new_parameter_symbols[p] = symbol_type(p.name)
        substituted_symbol_expr = self._symbol_expr.subs(symbol_map)
        return ParameterExpression(new_parameter_symbols, substituted_symbol_expr)

    def _raise_if_passed_unknown_parameters(self, parameters):
        if False:
            while True:
                i = 10
        unknown_parameters = parameters - self.parameters
        if unknown_parameters:
            raise CircuitError('Cannot bind Parameters ({}) not present in expression.'.format([str(p) for p in unknown_parameters]))

    def _raise_if_passed_nan(self, parameter_values):
        if False:
            while True:
                i = 10
        nan_parameter_values = {p: v for (p, v) in parameter_values.items() if not isinstance(v, numbers.Number)}
        if nan_parameter_values:
            raise CircuitError(f'Expression cannot bind non-numeric values ({nan_parameter_values})')

    def _raise_if_parameter_names_conflict(self, inbound_parameters, outbound_parameters=None):
        if False:
            return 10
        if outbound_parameters is None:
            outbound_parameters = set()
            outbound_names = {}
        else:
            outbound_names = {p.name: p for p in outbound_parameters}
        inbound_names = inbound_parameters
        conflicting_names = []
        for (name, param) in inbound_names.items():
            if name in self._names and name not in outbound_names:
                if param != self._names[name]:
                    conflicting_names.append(name)
        if conflicting_names:
            raise CircuitError(f'Name conflict applying operation for parameters: {conflicting_names}')

    def _apply_operation(self, operation: Callable, other: ParameterValueType, reflected: bool=False) -> 'ParameterExpression':
        if False:
            while True:
                i = 10
        'Base method implementing math operations between Parameters and\n        either a constant or a second ParameterExpression.\n\n        Args:\n            operation: One of operator.{add,sub,mul,truediv}.\n            other: The second argument to be used with self in operation.\n            reflected: Optional - The default ordering is "self operator other".\n                       If reflected is True, this is switched to "other operator self".\n                       For use in e.g. __radd__, ...\n\n        Raises:\n            CircuitError:\n                - If parameter_map contains Parameters outside those in self.\n                - If the replacement Parameters in parameter_map would result in\n                  a name conflict in the generated expression.\n\n        Returns:\n            A new expression describing the result of the operation.\n        '
        self_expr = self._symbol_expr
        if isinstance(other, ParameterExpression):
            self._raise_if_parameter_names_conflict(other._names)
            parameter_symbols = {**self._parameter_symbols, **other._parameter_symbols}
            other_expr = other._symbol_expr
        elif isinstance(other, numbers.Number) and numpy.isfinite(other):
            parameter_symbols = self._parameter_symbols.copy()
            other_expr = other
        else:
            return NotImplemented
        if reflected:
            expr = operation(other_expr, self_expr)
        else:
            expr = operation(self_expr, other_expr)
        out_expr = ParameterExpression(parameter_symbols, expr)
        out_expr._name_map = self._names.copy()
        if isinstance(other, ParameterExpression):
            out_expr._names.update(other._names.copy())
        return out_expr

    def gradient(self, param) -> Union['ParameterExpression', complex]:
        if False:
            print('Hello World!')
        'Get the derivative of a parameter expression w.r.t. a specified parameter expression.\n\n        Args:\n            param (Parameter): Parameter w.r.t. which we want to take the derivative\n\n        Returns:\n            ParameterExpression representing the gradient of param_expr w.r.t. param\n            or complex or float number\n        '
        if param not in self._parameter_symbols.keys():
            return 0.0
        key = self._parameter_symbols[param]
        if _optionals.HAS_SYMENGINE:
            import symengine
            expr_grad = symengine.Derivative(self._symbol_expr, key)
        else:
            from sympy import Derivative
            expr_grad = Derivative(self._symbol_expr, key).doit()
        parameter_symbols = {}
        for (parameter, symbol) in self._parameter_symbols.items():
            if symbol in expr_grad.free_symbols:
                parameter_symbols[parameter] = symbol
        if len(parameter_symbols) > 0:
            return ParameterExpression(parameter_symbols, expr=expr_grad)
        expr_grad_cplx = complex(expr_grad)
        if expr_grad_cplx.imag != 0:
            return expr_grad_cplx
        else:
            return float(expr_grad)

    def __add__(self, other):
        if False:
            print('Hello World!')
        return self._apply_operation(operator.add, other)

    def __radd__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self._apply_operation(operator.add, other, reflected=True)

    def __sub__(self, other):
        if False:
            while True:
                i = 10
        return self._apply_operation(operator.sub, other)

    def __rsub__(self, other):
        if False:
            i = 10
            return i + 15
        return self._apply_operation(operator.sub, other, reflected=True)

    def __mul__(self, other):
        if False:
            while True:
                i = 10
        return self._apply_operation(operator.mul, other)

    def __neg__(self):
        if False:
            while True:
                i = 10
        return self._apply_operation(operator.mul, -1.0)

    def __rmul__(self, other):
        if False:
            i = 10
            return i + 15
        return self._apply_operation(operator.mul, other, reflected=True)

    def __truediv__(self, other):
        if False:
            print('Hello World!')
        if other == 0:
            raise ZeroDivisionError('Division of a ParameterExpression by zero.')
        return self._apply_operation(operator.truediv, other)

    def __rtruediv__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self._apply_operation(operator.truediv, other, reflected=True)

    def _call(self, ufunc):
        if False:
            for i in range(10):
                print('nop')
        return ParameterExpression(self._parameter_symbols, ufunc(self._symbol_expr))

    def sin(self):
        if False:
            print('Hello World!')
        'Sine of a ParameterExpression'
        if _optionals.HAS_SYMENGINE:
            import symengine
            return self._call(symengine.sin)
        else:
            from sympy import sin as _sin
            return self._call(_sin)

    def cos(self):
        if False:
            for i in range(10):
                print('nop')
        'Cosine of a ParameterExpression'
        if _optionals.HAS_SYMENGINE:
            import symengine
            return self._call(symengine.cos)
        else:
            from sympy import cos as _cos
            return self._call(_cos)

    def tan(self):
        if False:
            i = 10
            return i + 15
        'Tangent of a ParameterExpression'
        if _optionals.HAS_SYMENGINE:
            import symengine
            return self._call(symengine.tan)
        else:
            from sympy import tan as _tan
            return self._call(_tan)

    def arcsin(self):
        if False:
            return 10
        'Arcsin of a ParameterExpression'
        if _optionals.HAS_SYMENGINE:
            import symengine
            return self._call(symengine.asin)
        else:
            from sympy import asin as _asin
            return self._call(_asin)

    def arccos(self):
        if False:
            i = 10
            return i + 15
        'Arccos of a ParameterExpression'
        if _optionals.HAS_SYMENGINE:
            import symengine
            return self._call(symengine.acos)
        else:
            from sympy import acos as _acos
            return self._call(_acos)

    def arctan(self):
        if False:
            return 10
        'Arctan of a ParameterExpression'
        if _optionals.HAS_SYMENGINE:
            import symengine
            return self._call(symengine.atan)
        else:
            from sympy import atan as _atan
            return self._call(_atan)

    def exp(self):
        if False:
            return 10
        'Exponential of a ParameterExpression'
        if _optionals.HAS_SYMENGINE:
            import symengine
            return self._call(symengine.exp)
        else:
            from sympy import exp as _exp
            return self._call(_exp)

    def log(self):
        if False:
            while True:
                i = 10
        'Logarithm of a ParameterExpression'
        if _optionals.HAS_SYMENGINE:
            import symengine
            return self._call(symengine.log)
        else:
            from sympy import log as _log
            return self._call(_log)

    def sign(self):
        if False:
            while True:
                i = 10
        'Sign of a ParameterExpression'
        if _optionals.HAS_SYMENGINE:
            import symengine
            return self._call(symengine.sign)
        else:
            from sympy import sign as _sign
            return self._call(_sign)

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'{self.__class__.__name__}({str(self)})'

    def __str__(self):
        if False:
            while True:
                i = 10
        from sympy import sympify, sstr
        return sstr(sympify(self._symbol_expr), full_prec=False)

    def __complex__(self):
        if False:
            while True:
                i = 10
        try:
            return complex(self._symbol_expr)
        except (TypeError, RuntimeError) as exc:
            if self.parameters:
                raise TypeError('ParameterExpression with unbound parameters ({}) cannot be cast to a complex.'.format(self.parameters)) from None
            raise TypeError('could not cast expression to complex') from exc

    def __float__(self):
        if False:
            i = 10
            return i + 15
        try:
            return float(self._symbol_expr)
        except (TypeError, RuntimeError) as exc:
            if self.parameters:
                raise TypeError('ParameterExpression with unbound parameters ({}) cannot be cast to a float.'.format(self.parameters)) from None
            try:
                cval = complex(self)
                if cval.imag == 0.0:
                    return cval.real
            except TypeError:
                pass
            raise TypeError('could not cast expression to float') from exc

    def __int__(self):
        if False:
            return 10
        try:
            return int(self._symbol_expr)
        except (TypeError, RuntimeError) as exc:
            if self.parameters:
                raise TypeError('ParameterExpression with unbound parameters ({}) cannot be cast to an int.'.format(self.parameters)) from None
            raise TypeError('could not cast expression to int') from exc

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash((self._parameter_keys, self._symbol_expr))

    def __copy__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __deepcopy__(self, memo=None):
        if False:
            return 10
        return self

    def __abs__(self):
        if False:
            print('Hello World!')
        'Absolute of a ParameterExpression'
        if _optionals.HAS_SYMENGINE:
            import symengine
            return self._call(symengine.Abs)
        else:
            from sympy import Abs as _abs
            return self._call(_abs)

    def abs(self):
        if False:
            print('Hello World!')
        'Absolute of a ParameterExpression'
        return self.__abs__()

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        'Check if this parameter expression is equal to another parameter expression\n           or a fixed value (only if this is a bound expression).\n        Args:\n            other (ParameterExpression or a number):\n                Parameter expression or numeric constant used for comparison\n        Returns:\n            bool: result of the comparison\n        '
        if isinstance(other, ParameterExpression):
            if self.parameters != other.parameters:
                return False
            if _optionals.HAS_SYMENGINE:
                from sympy import sympify
                return sympify(self._symbol_expr).equals(sympify(other._symbol_expr))
            else:
                return self._symbol_expr.equals(other._symbol_expr)
        elif isinstance(other, numbers.Number):
            return len(self.parameters) == 0 and complex(self._symbol_expr) == other
        return False

    def is_real(self):
        if False:
            i = 10
            return i + 15
        'Return whether the expression is real'
        if _optionals.HAS_SYMENGINE and self._symbol_expr.is_real is None:
            symbol_expr = self._symbol_expr.evalf()
        else:
            symbol_expr = self._symbol_expr
        if not symbol_expr.is_real and symbol_expr.is_real is not None:
            if _optionals.HAS_SYMENGINE:
                if symbol_expr.imag == 0.0:
                    return True
            return False
        return symbol_expr.is_real

    def sympify(self):
        if False:
            for i in range(10):
                print('nop')
        'Return symbolic expression as a raw Sympy or Symengine object.\n\n        Symengine is used preferentially; if both are available, the result will always be a\n        ``symengine`` object.  Symengine is a separate library but has integration with Sympy.\n\n        .. note::\n\n            This is for interoperability only.  Qiskit will not accept or work with raw Sympy or\n            Symegine expressions in its parameters, because they do not contain the tracking\n            information used in circuit-parameter binding and assignment.\n        '
        return self._symbol_expr
ParameterValueType = Union[ParameterExpression, float]