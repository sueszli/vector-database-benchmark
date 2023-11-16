from __future__ import annotations
import logging
import traceback
from typing import Any
from nni.mutable import MutableExpression, Categorical, Numerical
_logger = logging.getLogger(__name__)

def conclude_assumptions(values: list[int | float]) -> dict[str, bool]:
    if False:
        for i in range(10):
            print('nop')
    'Conclude some sympy assumptions based on the examples in values.\n\n    Support assumptions are: positive, negative, nonpositive, nonnegative,\n    zero, nonzero, odd, even, real.\n    '
    if not values:
        return {}
    assumptions = {}
    assumptions['real'] = all((isinstance(v, (float, int)) for v in values))
    if not assumptions['real']:
        return assumptions
    assumptions['integer'] = all((isinstance(v, int) for v in values))
    if all((v > 0 for v in values)):
        assumptions['positive'] = True
    if all((v < 0 for v in values)):
        assumptions['negative'] = True
    if all((v >= 0 for v in values)):
        assumptions['nonnegative'] = True
    if all((v <= 0 for v in values)):
        assumptions['nonpositive'] = True
    if all((v == 0 for v in values)):
        assumptions['zero'] = True
    if all((v != 0 for v in values)):
        assumptions['nonzero'] = True
    if not assumptions['integer']:
        return assumptions
    if all((v % 2 == 0 for v in values)):
        assumptions['even'] = True
    if all((v % 2 == 1 for v in values)):
        assumptions['odd'] = True
    return assumptions
_seen_errors = set()

def expression_simplification(expression: MutableExpression):
    if False:
        for i in range(10):
            print('nop')
    try:
        from sympy import Symbol, Expr, lambdify, simplify
    except ImportError:
        _logger.warning('sympy is not installed, give up expression simplification.')
        return expression
    mutables = expression.simplify()
    mutable_substitutes: dict[str, Symbol | Expr] = {}
    inverse_substitutes: dict[Symbol | Expr, MutableExpression] = {}
    for (name, mutable) in mutables.items():
        if isinstance(mutable, Categorical):
            assumptions = conclude_assumptions(mutable.values)
            if not assumptions.get('real', False):
                _logger.warning('Expression simplification only supports categorical mutables with numerical choices, but got %r. Give up.', mutable)
            odd = assumptions.pop('odd', False)
            if odd:
                if not assumptions.get('positive', False):
                    if assumptions.get('nonnegative', False):
                        assumptions['nonpositive'] = True
                    assumptions.pop('nonzero', None)
                symbol = Symbol(name, **assumptions)
                mutable_substitutes[name] = symbol * 2 - 1
                inverse_substitutes[symbol] = (mutable + 1) // 2
            else:
                symbol = Symbol(name, **assumptions)
                mutable_substitutes[name] = symbol
                inverse_substitutes[symbol] = mutable
        elif isinstance(mutable, Numerical):
            symbol = Symbol(name, real=True)
            mutable_substitutes[name] = symbol
            inverse_substitutes[symbol] = mutable
        else:
            _logger.warning('Expression simplification only supports categorical and numerical mutables, but got %s in expression. Give up.', type(mutable))
            return expression
    try:
        sym_expression = expression.evaluate(mutable_substitutes)
        simplified_sym_expression = simplify(sym_expression)
        simplified_fn = lambdify(list(inverse_substitutes.keys()), simplified_sym_expression)
        simplified_expr = simplified_fn(*inverse_substitutes.values())
        expected_type = type(expression.default())
        actual_type = type(simplified_expr.default() if isinstance(simplified_expr, MutableExpression) else simplified_expr)
        if actual_type != expected_type:
            if expected_type == int:
                simplified_expr = round(simplified_expr)
            elif expected_type == float:
                simplified_expr = MutableExpression.to_float(simplified_expr)
            else:
                _logger.warning('Simplified expression is of type %s, but expected type is %s. Cannot convert.', actual_type, expected_type)
                return expression
    except Exception as e:
        error_repr = repr(e)
        if error_repr not in _seen_errors:
            _seen_errors.add(error_repr)
            _logger.warning('Expression simplification failed: %s. Give up.\nExpression: %s\n%s', error_repr, expression, traceback.format_exc())
        else:
            pass
        return expression
    return simplified_expr

def recursive_simplification(obj: Any) -> Any:
    if False:
        print('Hello World!')
    'Simplify all expressions in obj recursively.'
    from .shape import MutableShape
    if isinstance(obj, MutableExpression):
        return expression_simplification(obj)
    elif isinstance(obj, MutableShape):
        return MutableShape(*[recursive_simplification(v) for v in obj])
    elif isinstance(obj, dict):
        return {k: recursive_simplification(v) for (k, v) in obj.items()}
    elif isinstance(obj, list):
        return [recursive_simplification(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple((recursive_simplification(v) for v in obj))
    return obj