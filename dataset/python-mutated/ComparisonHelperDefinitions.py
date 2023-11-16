""" Shared definitions of what comparison operation helpers are available.

These are functions to work with helper names, as well as sets of functions to
generate specialized code variants for.

Note: These are ordered, so we can define the order they are created in the
code generation of specialized helpers, as this set is used for input there
too.

"""
from nuitka.containers.OrderedSets import buildOrderedSet
rich_comparison_codes = {'Lt': 'LT', 'LtE': 'LE', 'Eq': 'EQ', 'NotEq': 'NE', 'Gt': 'GT', 'GtE': 'GE'}
rich_comparison_subset_codes = {'Lt': 'LT', 'LtE': 'LE', 'Eq': 'EQ'}

def _makeDefaultOps():
    if False:
        while True:
            i = 10
    for comparator in rich_comparison_codes.values():
        yield ('RICH_COMPARE_%s_OBJECT_OBJECT_OBJECT' % comparator)
        yield ('RICH_COMPARE_%s_NBOOL_OBJECT_OBJECT' % comparator)

def _makeTypeOps(type_name, may_raise_same_type, shortcut=False):
    if False:
        return 10
    for result_part in ('OBJECT', 'CBOOL', 'NBOOL'):
        for comparator in rich_comparison_codes.values():
            if result_part == 'CBOOL':
                continue
            yield ('RICH_COMPARE_%s_%s_OBJECT_%s' % (comparator, result_part, type_name))
            yield ('RICH_COMPARE_%s_%s_%s_OBJECT' % (comparator, result_part, type_name))
        if may_raise_same_type and result_part == 'CBOOL':
            continue
        if not may_raise_same_type and result_part == 'NBOOL':
            continue
        for comparator in rich_comparison_codes.values() if not shortcut else rich_comparison_subset_codes.values():
            yield ('RICH_COMPARE_%s_%s_%s_%s' % (comparator, result_part, type_name, type_name))

def _makeFriendOps(type_name1, type_name2, may_raise):
    if False:
        return 10
    assert type_name1 != type_name2
    for result_part in ('OBJECT', 'CBOOL', 'NBOOL'):
        if not may_raise:
            if result_part == 'NBOOL':
                continue
        for comparator in rich_comparison_codes.values():
            yield ('RICH_COMPARE_%s_%s_%s_%s' % (comparator, result_part, type_name1, type_name2))
specialized_cmp_helpers_set = buildOrderedSet(_makeDefaultOps(), _makeTypeOps('STR', may_raise_same_type=False, shortcut=True), _makeTypeOps('UNICODE', may_raise_same_type=False, shortcut=True), _makeTypeOps('BYTES', may_raise_same_type=False, shortcut=True), _makeTypeOps('INT', may_raise_same_type=False, shortcut=True), _makeTypeOps('LONG', may_raise_same_type=False, shortcut=True), _makeTypeOps('FLOAT', may_raise_same_type=False, shortcut=True), _makeTypeOps('TUPLE', may_raise_same_type=True), _makeTypeOps('LIST', may_raise_same_type=True), _makeFriendOps('LONG', 'INT', may_raise=False), _makeFriendOps('INT', 'CLONG', may_raise=False), _makeFriendOps('LONG', 'DIGIT', may_raise=False), _makeFriendOps('FLOAT', 'CFLOAT', may_raise=False))
_non_specialized_cmp_helpers_set = set()

def getSpecializedComparisonOperations():
    if False:
        return 10
    return specialized_cmp_helpers_set

def getNonSpecializedComparisonOperations():
    if False:
        return 10
    return _non_specialized_cmp_helpers_set