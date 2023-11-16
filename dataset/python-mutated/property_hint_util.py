"""Common utilities for LinearOperator property hints."""

def combined_commuting_self_adjoint_hint(operator_a, operator_b):
    if False:
        i = 10
        return i + 15
    'Get combined hint for self-adjoint-ness.'
    if operator_a.is_self_adjoint and operator_b.is_self_adjoint:
        return True
    if operator_a.is_self_adjoint is True and operator_b.is_self_adjoint is False or (operator_a.is_self_adjoint is False and operator_b.is_self_adjoint is True):
        return False
    return None

def is_square(operator_a, operator_b):
    if False:
        i = 10
        return i + 15
    'Return a hint to whether the composition is square.'
    if operator_a.is_square and operator_b.is_square:
        return True
    if operator_a.is_square is False and operator_b.is_square is False:
        m = operator_a.range_dimension
        l = operator_b.domain_dimension
        if m is not None and l is not None:
            return m == l
    if operator_a.is_square != operator_b.is_square and (operator_a.is_square is not None and operator_b.is_square is not None):
        return False
    return None

def combined_commuting_positive_definite_hint(operator_a, operator_b):
    if False:
        i = 10
        return i + 15
    'Get combined PD hint for compositions.'
    if operator_a.is_positive_definite is True and operator_a.is_self_adjoint is True and (operator_b.is_positive_definite is True) and (operator_b.is_self_adjoint is True):
        return True
    return None

def combined_non_singular_hint(operator_a, operator_b):
    if False:
        i = 10
        return i + 15
    'Get combined hint for when .'
    if operator_a.is_non_singular is False or operator_b.is_non_singular is False:
        return False
    return operator_a.is_non_singular and operator_b.is_non_singular