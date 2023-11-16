"""Functions which help end users define customize node_match and
edge_match functions to use during isomorphism checks.
"""
import math
import types
from itertools import permutations
__all__ = ['categorical_node_match', 'categorical_edge_match', 'categorical_multiedge_match', 'numerical_node_match', 'numerical_edge_match', 'numerical_multiedge_match', 'generic_node_match', 'generic_edge_match', 'generic_multiedge_match']

def copyfunc(f, name=None):
    if False:
        while True:
            i = 10
    'Returns a deepcopy of a function.'
    return types.FunctionType(f.__code__, f.__globals__, name or f.__name__, f.__defaults__, f.__closure__)

def allclose(x, y, rtol=1e-05, atol=1e-08):
    if False:
        i = 10
        return i + 15
    'Returns True if x and y are sufficiently close, elementwise.\n\n    Parameters\n    ----------\n    rtol : float\n        The relative error tolerance.\n    atol : float\n        The absolute error tolerance.\n\n    '
    return all((math.isclose(xi, yi, rel_tol=rtol, abs_tol=atol) for (xi, yi) in zip(x, y)))
categorical_doc = '\nReturns a comparison function for a categorical node attribute.\n\nThe value(s) of the attr(s) must be hashable and comparable via the ==\noperator since they are placed into a set([]) object.  If the sets from\nG1 and G2 are the same, then the constructed function returns True.\n\nParameters\n----------\nattr : string | list\n    The categorical node attribute to compare, or a list of categorical\n    node attributes to compare.\ndefault : value | list\n    The default value for the categorical node attribute, or a list of\n    default values for the categorical node attributes.\n\nReturns\n-------\nmatch : function\n    The customized, categorical `node_match` function.\n\nExamples\n--------\n>>> import networkx.algorithms.isomorphism as iso\n>>> nm = iso.categorical_node_match("size", 1)\n>>> nm = iso.categorical_node_match(["color", "size"], ["red", 2])\n\n'

def categorical_node_match(attr, default):
    if False:
        while True:
            i = 10
    if isinstance(attr, str):

        def match(data1, data2):
            if False:
                print('Hello World!')
            return data1.get(attr, default) == data2.get(attr, default)
    else:
        attrs = list(zip(attr, default))

        def match(data1, data2):
            if False:
                while True:
                    i = 10
            return all((data1.get(attr, d) == data2.get(attr, d) for (attr, d) in attrs))
    return match
categorical_edge_match = copyfunc(categorical_node_match, 'categorical_edge_match')

def categorical_multiedge_match(attr, default):
    if False:
        while True:
            i = 10
    if isinstance(attr, str):

        def match(datasets1, datasets2):
            if False:
                return 10
            values1 = {data.get(attr, default) for data in datasets1.values()}
            values2 = {data.get(attr, default) for data in datasets2.values()}
            return values1 == values2
    else:
        attrs = list(zip(attr, default))

        def match(datasets1, datasets2):
            if False:
                while True:
                    i = 10
            values1 = set()
            for data1 in datasets1.values():
                x = tuple((data1.get(attr, d) for (attr, d) in attrs))
                values1.add(x)
            values2 = set()
            for data2 in datasets2.values():
                x = tuple((data2.get(attr, d) for (attr, d) in attrs))
                values2.add(x)
            return values1 == values2
    return match
categorical_node_match.__doc__ = categorical_doc
categorical_edge_match.__doc__ = categorical_doc.replace('node', 'edge')
tmpdoc = categorical_doc.replace('node', 'edge')
tmpdoc = tmpdoc.replace('categorical_edge_match', 'categorical_multiedge_match')
categorical_multiedge_match.__doc__ = tmpdoc
numerical_doc = '\nReturns a comparison function for a numerical node attribute.\n\nThe value(s) of the attr(s) must be numerical and sortable.  If the\nsorted list of values from G1 and G2 are the same within some\ntolerance, then the constructed function returns True.\n\nParameters\n----------\nattr : string | list\n    The numerical node attribute to compare, or a list of numerical\n    node attributes to compare.\ndefault : value | list\n    The default value for the numerical node attribute, or a list of\n    default values for the numerical node attributes.\nrtol : float\n    The relative error tolerance.\natol : float\n    The absolute error tolerance.\n\nReturns\n-------\nmatch : function\n    The customized, numerical `node_match` function.\n\nExamples\n--------\n>>> import networkx.algorithms.isomorphism as iso\n>>> nm = iso.numerical_node_match("weight", 1.0)\n>>> nm = iso.numerical_node_match(["weight", "linewidth"], [0.25, 0.5])\n\n'

def numerical_node_match(attr, default, rtol=1e-05, atol=1e-08):
    if False:
        return 10
    if isinstance(attr, str):

        def match(data1, data2):
            if False:
                while True:
                    i = 10
            return math.isclose(data1.get(attr, default), data2.get(attr, default), rel_tol=rtol, abs_tol=atol)
    else:
        attrs = list(zip(attr, default))

        def match(data1, data2):
            if False:
                while True:
                    i = 10
            values1 = [data1.get(attr, d) for (attr, d) in attrs]
            values2 = [data2.get(attr, d) for (attr, d) in attrs]
            return allclose(values1, values2, rtol=rtol, atol=atol)
    return match
numerical_edge_match = copyfunc(numerical_node_match, 'numerical_edge_match')

def numerical_multiedge_match(attr, default, rtol=1e-05, atol=1e-08):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(attr, str):

        def match(datasets1, datasets2):
            if False:
                return 10
            values1 = sorted((data.get(attr, default) for data in datasets1.values()))
            values2 = sorted((data.get(attr, default) for data in datasets2.values()))
            return allclose(values1, values2, rtol=rtol, atol=atol)
    else:
        attrs = list(zip(attr, default))

        def match(datasets1, datasets2):
            if False:
                print('Hello World!')
            values1 = []
            for data1 in datasets1.values():
                x = tuple((data1.get(attr, d) for (attr, d) in attrs))
                values1.append(x)
            values2 = []
            for data2 in datasets2.values():
                x = tuple((data2.get(attr, d) for (attr, d) in attrs))
                values2.append(x)
            values1.sort()
            values2.sort()
            for (xi, yi) in zip(values1, values2):
                if not allclose(xi, yi, rtol=rtol, atol=atol):
                    return False
            else:
                return True
    return match
numerical_node_match.__doc__ = numerical_doc
numerical_edge_match.__doc__ = numerical_doc.replace('node', 'edge')
tmpdoc = numerical_doc.replace('node', 'edge')
tmpdoc = tmpdoc.replace('numerical_edge_match', 'numerical_multiedge_match')
numerical_multiedge_match.__doc__ = tmpdoc
generic_doc = '\nReturns a comparison function for a generic attribute.\n\nThe value(s) of the attr(s) are compared using the specified\noperators. If all the attributes are equal, then the constructed\nfunction returns True.\n\nParameters\n----------\nattr : string | list\n    The node attribute to compare, or a list of node attributes\n    to compare.\ndefault : value | list\n    The default value for the node attribute, or a list of\n    default values for the node attributes.\nop : callable | list\n    The operator to use when comparing attribute values, or a list\n    of operators to use when comparing values for each attribute.\n\nReturns\n-------\nmatch : function\n    The customized, generic `node_match` function.\n\nExamples\n--------\n>>> from operator import eq\n>>> from math import isclose\n>>> from networkx.algorithms.isomorphism import generic_node_match\n>>> nm = generic_node_match("weight", 1.0, isclose)\n>>> nm = generic_node_match("color", "red", eq)\n>>> nm = generic_node_match(["weight", "color"], [1.0, "red"], [isclose, eq])\n\n'

def generic_node_match(attr, default, op):
    if False:
        print('Hello World!')
    if isinstance(attr, str):

        def match(data1, data2):
            if False:
                i = 10
                return i + 15
            return op(data1.get(attr, default), data2.get(attr, default))
    else:
        attrs = list(zip(attr, default, op))

        def match(data1, data2):
            if False:
                return 10
            for (attr, d, operator) in attrs:
                if not operator(data1.get(attr, d), data2.get(attr, d)):
                    return False
            else:
                return True
    return match
generic_edge_match = copyfunc(generic_node_match, 'generic_edge_match')

def generic_multiedge_match(attr, default, op):
    if False:
        for i in range(10):
            print('nop')
    'Returns a comparison function for a generic attribute.\n\n    The value(s) of the attr(s) are compared using the specified\n    operators. If all the attributes are equal, then the constructed\n    function returns True. Potentially, the constructed edge_match\n    function can be slow since it must verify that no isomorphism\n    exists between the multiedges before it returns False.\n\n    Parameters\n    ----------\n    attr : string | list\n        The edge attribute to compare, or a list of node attributes\n        to compare.\n    default : value | list\n        The default value for the edge attribute, or a list of\n        default values for the edgeattributes.\n    op : callable | list\n        The operator to use when comparing attribute values, or a list\n        of operators to use when comparing values for each attribute.\n\n    Returns\n    -------\n    match : function\n        The customized, generic `edge_match` function.\n\n    Examples\n    --------\n    >>> from operator import eq\n    >>> from math import isclose\n    >>> from networkx.algorithms.isomorphism import generic_node_match\n    >>> nm = generic_node_match("weight", 1.0, isclose)\n    >>> nm = generic_node_match("color", "red", eq)\n    >>> nm = generic_node_match(["weight", "color"], [1.0, "red"], [isclose, eq])\n    ...\n\n    '
    if isinstance(attr, str):
        attr = [attr]
        default = [default]
        op = [op]
    attrs = list(zip(attr, default))

    def match(datasets1, datasets2):
        if False:
            i = 10
            return i + 15
        values1 = []
        for data1 in datasets1.values():
            x = tuple((data1.get(attr, d) for (attr, d) in attrs))
            values1.append(x)
        values2 = []
        for data2 in datasets2.values():
            x = tuple((data2.get(attr, d) for (attr, d) in attrs))
            values2.append(x)
        for vals2 in permutations(values2):
            for (xi, yi) in zip(values1, vals2):
                if not all(map(lambda x, y, z: z(x, y), xi, yi, op)):
                    break
            else:
                return True
        else:
            return False
    return match
generic_node_match.__doc__ = generic_doc
generic_edge_match.__doc__ = generic_doc.replace('node', 'edge')