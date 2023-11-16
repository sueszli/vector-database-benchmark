import b2.build.feature
feature = b2.build.feature
from b2.util.utility import *
from b2.util import is_iterable_typed
import b2.build.property_set as property_set

def expand_no_defaults(property_sets):
    if False:
        for i in range(10):
            print('nop')
    " Expand the given build request by combining all property_sets which don't\n        specify conflicting non-free features.\n    "
    assert is_iterable_typed(property_sets, property_set.PropertySet)
    expanded_property_sets = [ps.expand_subfeatures() for ps in property_sets]
    product = __x_product(expanded_property_sets)
    return [property_set.create(p) for p in product]

def __x_product(property_sets):
    if False:
        print('Hello World!')
    ' Return the cross-product of all elements of property_sets, less any\n        that would contain conflicting values for single-valued features.\n    '
    assert is_iterable_typed(property_sets, property_set.PropertySet)
    x_product_seen = set()
    return __x_product_aux(property_sets, x_product_seen)[0]

def __x_product_aux(property_sets, seen_features):
    if False:
        while True:
            i = 10
    'Returns non-conflicting combinations of property sets.\n\n    property_sets is a list of PropertySet instances. seen_features is a set of Property\n    instances.\n\n    Returns a tuple of:\n    - list of lists of Property instances, such that within each list, no two Property instance\n    have the same feature, and no Property is for feature in seen_features.\n    - set of features we saw in property_sets\n    '
    assert is_iterable_typed(property_sets, property_set.PropertySet)
    assert isinstance(seen_features, set)
    if not property_sets:
        return ([], set())
    properties = property_sets[0].all()
    these_features = set()
    for p in property_sets[0].non_free():
        these_features.add(p.feature)
    if these_features & seen_features:
        (inner_result, inner_seen) = __x_product_aux(property_sets[1:], seen_features)
        return (inner_result, inner_seen | these_features)
    else:
        result = []
        (inner_result, inner_seen) = __x_product_aux(property_sets[1:], seen_features | these_features)
        if inner_result:
            for inner in inner_result:
                result.append(properties + inner)
        else:
            result.append(properties)
        if inner_seen & these_features:
            (inner_result2, inner_seen2) = __x_product_aux(property_sets[1:], seen_features)
            result.extend(inner_result2)
        return (result, inner_seen | these_features)

def looks_like_implicit_value(v):
    if False:
        return 10
    "Returns true if 'v' is either implicit value, or\n    the part before the first '-' symbol is implicit value."
    assert isinstance(v, basestring)
    if feature.is_implicit_value(v):
        return 1
    else:
        split = v.split('-')
        if feature.is_implicit_value(split[0]):
            return 1
    return 0

def from_command_line(command_line):
    if False:
        i = 10
        return i + 15
    'Takes the command line tokens (such as taken from ARGV rule)\n    and constructs build request from it. Returns a list of two\n    lists. First is the set of targets specified in the command line,\n    and second is the set of requested build properties.'
    assert is_iterable_typed(command_line, basestring)
    targets = []
    properties = []
    for e in command_line:
        if e[:1] != '-':
            if e.find('=') != -1 or looks_like_implicit_value(e.split('/')[0]):
                properties.append(e)
            elif e:
                targets.append(e)
    return [targets, properties]

def convert_command_line_element(e):
    if False:
        for i in range(10):
            print('nop')
    assert isinstance(e, basestring)
    result = None
    parts = e.split('/')
    for p in parts:
        m = p.split('=')
        if len(m) > 1:
            feature = m[0]
            values = m[1].split(',')
            lresult = ['<%s>%s' % (feature, v) for v in values]
        else:
            lresult = p.split(',')
        if p.find('-') == -1:
            pass
        if not result:
            result = lresult
        else:
            result = [e1 + '/' + e2 for e1 in result for e2 in lresult]
    return [property_set.create(b2.build.feature.split(r)) for r in result]