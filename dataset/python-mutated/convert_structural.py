"""Solver for type equations."""
import itertools
import logging
from typing import AbstractSet, Dict
from pytype.pytd import booleq
from pytype.pytd import escape
from pytype.pytd import optimize
from pytype.pytd import pytd
from pytype.pytd import pytd_utils
from pytype.pytd import transforms
from pytype.pytd import type_match
from pytype.pytd import visitors
log = logging.getLogger(__name__)
MAX_DEPTH = 1
is_unknown = type_match.is_unknown
is_partial = escape.is_partial
is_complete = escape.is_complete

class FlawedQuery(Exception):
    """Thrown if there is a fundamental flaw in the query."""

class TypeSolver:
    """Class for solving ~unknowns in type inference results."""

    def __init__(self, ast, builtins, protocols):
        if False:
            return 10
        self.ast = ast
        self.builtins = builtins
        self.protocols = protocols

    def match_unknown_against_protocol(self, matcher, solver, unknown, complete):
        if False:
            while True:
                i = 10
        'Given an ~unknown, match it against a class.\n\n    Args:\n      matcher: An instance of pytd.type_match.TypeMatch.\n      solver: An instance of pytd.booleq.Solver.\n      unknown: The unknown class to match\n      complete: A complete class to match against. (E.g. a built-in or a user\n        defined class)\n    Returns:\n      An instance of pytd.booleq.BooleanTerm.\n    '
        assert is_unknown(unknown)
        assert is_complete(complete)
        type_params = {p.type_param: matcher.type_parameter(unknown, complete, p) for p in complete.template}
        subst = type_params.copy()
        implication = matcher.match_Protocol_against_Unknown(complete, unknown, subst)
        if implication is not booleq.FALSE and type_params:
            for param in type_params.values():
                solver.register_variable(param.name)
        solver.implies(booleq.Eq(unknown.name, complete.name), implication)

    def match_partial_against_complete(self, matcher, solver, partial, complete):
        if False:
            i = 10
            return i + 15
        'Match a partial class (call record) against a complete class.\n\n    Args:\n      matcher: An instance of pytd.type_match.TypeMatch.\n      solver: An instance of pytd.booleq.Solver.\n      partial: The partial class to match. The class name needs to be prefixed\n        with "~" - the rest of the name is typically the same as complete.name.\n      complete: A complete class to match against. (E.g. a built-in or a user\n        defined class)\n    Returns:\n      An instance of pytd.booleq.BooleanTerm.\n    Raises:\n      FlawedQuery: If this call record is incompatible with the builtin.\n    '
        assert is_partial(partial)
        assert is_complete(complete)
        subst = {p.type_param: pytd.AnythingType() for p in complete.template}
        formula = matcher.match_Class_against_Class(partial, complete, subst)
        if formula is booleq.FALSE:
            raise FlawedQuery(f'{partial.name} can never be {complete.name}')
        solver.always_true(formula)

    def match_call_record(self, matcher, solver, call_record, complete):
        if False:
            i = 10
            return i + 15
        'Match the record of a method call against the formal signature.'
        assert is_partial(call_record)
        assert is_complete(complete)
        formula = matcher.match_Function_against_Function(call_record, complete, {})
        if formula is booleq.FALSE:
            cartesian = call_record.Visit(visitors.ExpandSignatures())
            for signature in cartesian.signatures:
                formula = matcher.match_Signature_against_Function(signature, complete, {})
                if formula is booleq.FALSE:
                    faulty_signature = pytd_utils.Print(signature)
                    break
            else:
                faulty_signature = ''
            raise FlawedQuery('Bad call\n{}{}\nagainst:\n{}'.format(escape.unpack_partial(call_record.name), faulty_signature, pytd_utils.Print(complete)))
        solver.always_true(formula)

    def solve(self):
        if False:
            i = 10
            return i + 15
        'Solve the equations generated from the pytd.\n\n    Returns:\n      A dictionary (str->str), mapping unknown class names to known class names.\n    Raises:\n      AssertionError: If we detect an internal error.\n    '
        hierarchy = type_match.get_all_subclasses([self.ast, self.builtins])
        factory_protocols = type_match.TypeMatch(hierarchy)
        factory_partial = type_match.TypeMatch(hierarchy)
        solver_protocols = factory_protocols.solver
        solver_partial = factory_partial.solver
        unknown_classes = set()
        partial_classes = set()
        complete_classes = set()
        for cls in self.ast.classes:
            if is_unknown(cls):
                solver_protocols.register_variable(cls.name)
                solver_partial.register_variable(cls.name)
                unknown_classes.add(cls)
            elif is_partial(cls):
                partial_classes.add(cls)
            else:
                complete_classes.add(cls)
        protocol_classes_and_aliases = set(self.protocols.classes)
        for alias in self.protocols.aliases:
            if not isinstance(alias.type, pytd.AnythingType) and alias.name != 'protocols.Protocol':
                protocol_classes_and_aliases.add(alias.type.cls)
        for protocol in protocol_classes_and_aliases:
            for unknown in unknown_classes:
                self.match_unknown_against_protocol(factory_protocols, solver_protocols, unknown, protocol)
        for complete in complete_classes.union(self.builtins.classes):
            for partial in partial_classes:
                if escape.unpack_partial(partial.name) == complete.name:
                    self.match_partial_against_complete(factory_partial, solver_partial, partial, complete)
        partial_functions = set()
        complete_functions = set()
        for f in self.ast.functions:
            if is_partial(f):
                partial_functions.add(f)
            else:
                complete_functions.add(f)
        for partial in partial_functions:
            for complete in complete_functions.union(self.builtins.functions):
                if escape.unpack_partial(partial.name) == complete.name:
                    self.match_call_record(factory_partial, solver_partial, partial, complete)
        log.info('=========== Equations to solve =============\n%s', solver_protocols)
        log.info('=========== Equations to solve (end) =======')
        solved_protocols = solver_protocols.solve()
        log.info('=========== Call trace equations to solve =============\n%s', solver_partial)
        log.info('=========== Call trace equations to solve (end) =======')
        solved_partial = solver_partial.solve()
        merged_solution = {}
        for unknown in itertools.chain(solved_protocols, solved_partial):
            if unknown in solved_protocols and unknown in solved_partial:
                merged_solution[unknown] = solved_protocols[unknown].union(solved_partial[unknown])
                merged_solution[unknown].discard('?')
            elif unknown in solved_protocols:
                merged_solution[unknown] = solved_protocols[unknown]
            else:
                merged_solution[unknown] = solved_partial[unknown]
        return merged_solution

def solve(ast, builtins_pytd, protocols_pytd):
    if False:
        for i in range(10):
            print('nop')
    'Solve the unknowns in a pytd AST using the standard Python builtins.\n\n  Args:\n    ast: A pytd.TypeDeclUnit, containing classes named ~unknownXX.\n    builtins_pytd: A pytd for builtins.\n    protocols_pytd: A pytd for protocols.\n\n  Returns:\n    A tuple of (1) a dictionary (str->str) mapping unknown class names to known\n    class names and (2) a pytd.TypeDeclUnit of the complete classes in ast.\n  '
    builtins_pytd = transforms.RemoveMutableParameters(builtins_pytd)
    builtins_pytd = visitors.LookupClasses(builtins_pytd)
    protocols_pytd = visitors.LookupClasses(protocols_pytd)
    ast = visitors.LookupClasses(ast, builtins_pytd)
    return (TypeSolver(ast, builtins_pytd, protocols_pytd).solve(), extract_local(ast))

def extract_local(ast):
    if False:
        return 10
    'Extract all classes that are not unknowns of call records of builtins.'
    return pytd.TypeDeclUnit(name=ast.name, classes=tuple((cls for cls in ast.classes if is_complete(cls))), functions=tuple((f for f in ast.functions if is_complete(f))), constants=tuple((c for c in ast.constants if is_complete(c))), type_params=ast.type_params, aliases=ast.aliases)

def convert_string_type(string_type, unknown, mapping, global_lookup, depth=0):
    if False:
        while True:
            i = 10
    'Convert a string representing a type back to a pytd type.'
    try:
        cls = global_lookup.Lookup(string_type)
        base_type = pytd_utils.NamedOrClassType(cls.name, cls)
    except KeyError:
        cls = None
        base_type = pytd_utils.NamedOrClassType(string_type, cls)
    if cls and cls.template:
        parameters = []
        for t in cls.template:
            type_param_name = unknown + '.' + string_type + '.' + t.name
            if type_param_name in mapping and depth < MAX_DEPTH:
                string_type_params = mapping[type_param_name]
                parameters.append(convert_string_type_list(string_type_params, unknown, mapping, global_lookup, depth + 1))
            else:
                parameters.append(pytd.AnythingType())
        return pytd.GenericType(base_type, tuple(parameters))
    else:
        return base_type

def convert_string_type_list(types_as_string, unknown, mapping, global_lookup, depth=0):
    if False:
        while True:
            i = 10
    'Like convert_string_type, but operate on a list.'
    if not types_as_string or booleq.Solver.ANY_VALUE in types_as_string:
        return pytd.AnythingType()
    return pytd_utils.JoinTypes((convert_string_type(type_as_string, unknown, mapping, global_lookup, depth) for type_as_string in types_as_string))

def insert_solution(result, mapping, global_lookup):
    if False:
        for i in range(10):
            print('nop')
    'Replace ~unknown types in a pytd with the actual (solved) types.'
    subst = {unknown: convert_string_type_list(types_as_strings, unknown, mapping, global_lookup) for (unknown, types_as_strings) in mapping.items()}
    result = result.Visit(optimize.RenameUnknowns(subst))
    result = result.Visit(optimize.RemoveDuplicates())
    return result.Visit(visitors.ReplaceTypesByName(subst))

def convert_pytd(ast, builtins_pytd, protocols_pytd):
    if False:
        i = 10
        return i + 15
    'Convert pytd with unknowns (structural types) to one with nominal types.'
    builtins_pytd = builtins_pytd.Visit(visitors.ClassTypeToNamedType())
    (mapping, result) = solve(ast, builtins_pytd, protocols_pytd)
    log_info_mapping(mapping)
    lookup = pytd_utils.Concat(builtins_pytd, result)
    result = insert_solution(result, mapping, lookup)
    if log.isEnabledFor(logging.INFO):
        log.info('=========== solve result =============\n%s', pytd_utils.Print(result))
        log.info('=========== solve result (end) =============')
    return result

def log_info_mapping(mapping: Dict[str, AbstractSet[str]]) -> None:
    if False:
        return 10
    'Print a raw type mapping. For debugging.'
    if log.isEnabledFor(logging.DEBUG):
        cutoff = 12
        log.debug('=========== (possible types) ===========')
        for (unknown, possible_types) in sorted(mapping.items()):
            if len(possible_types) > cutoff:
                log.debug('%s can be   %s, ... (total: %d)', unknown, ', '.join(sorted(possible_types)[0:cutoff]), len(possible_types))
            else:
                log.debug('%s can be %s', unknown, ', '.join(sorted(possible_types)))
        log.debug('=========== (end of possible types) ===========')