from torch.fx.experimental.graph_gradual_typechecker import Refine
from torch.fx.tensor_type import TensorType
from torch.fx.experimental.unification import Var, unify

def infer_symbolic_types_single_pass(traced):
    if False:
        i = 10
        return i + 15
    '\n    Calls our symbolic inferencer once.\n    '
    r = Refine(traced)
    r.refine()
    mgu = unify_eq(r.constraints)
    substitute_all_types(traced.graph, mgu)

def infer_symbolic_types(traced):
    if False:
        return 10
    '\n    Calls our symbolic inferencer twice.\n    This is useful when one pass is not enough\n    to infer all the information such as the case\n    for braodcasting.\n    '
    r = Refine(traced)
    r.refine()
    mgu = unify_eq(r.constraints)
    substitute_all_types(traced.graph, mgu)
    r = Refine(traced)
    r.refine()
    mgu = unify_eq(r.constraints)
    substitute_all_types(traced.graph, mgu)
    r.symbolic_relations()

def convert_eq(list_of_eq):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert equality constraints in the right format\n    to be used by unification library.\n    '
    lhs = []
    rhs = []
    for eq in list_of_eq:
        lhs.append(eq.lhs)
        rhs.append(eq.rhs)
    return (tuple(lhs), tuple(rhs))

def unify_eq(list_of_eq):
    if False:
        for i in range(10):
            print('nop')
    '\n    Apply unification to a set of\n    equality constraints\n    '
    (lhs, rhs) = convert_eq(list_of_eq)
    return unify(lhs, rhs)

def substitute_solution_one_type(mapping, t):
    if False:
        for i in range(10):
            print('nop')
    '\n    Apply the most general unifier to a type\n    '
    if isinstance(t, Var):
        if t in mapping.keys():
            return mapping[t]
        else:
            return t
    elif isinstance(t, TensorType):
        new_type = []
        for typ in t.__args__:
            if typ in mapping.keys():
                new_type.append(mapping[typ])
            else:
                new_type.append(typ)
        return TensorType(tuple(new_type))
    elif isinstance(t, list):
        new_type = []
        for typ in t:
            new_type.append(substitute_solution_one_type(mapping, typ))
        return new_type
    elif isinstance(t, tuple):
        new_type = []
        for typ in t:
            new_type.append(substitute_solution_one_type(mapping, typ))
        return tuple(new_type)
    else:
        return t

def substitute_all_types(graph, mapping):
    if False:
        return 10
    '\n    Apply the most general unifier to all types in a graph\n    till reaching a fixed point. If the input and output graph\n    are the same, we converge.\n    '
    flag = True
    while flag:
        flag = False
        for k in mapping:
            old_mapping_val = mapping[k]
            if mapping[k] in mapping.keys():
                new_key = mapping[k]
                mapping[k] = mapping[new_key]
            if old_mapping_val != mapping[k]:
                flag = True
    for n in graph.nodes:
        n.type = substitute_solution_one_type(mapping, n.type)

def check_for_type_equality(g1, g2):
    if False:
        print('Hello World!')
    '\n    A check equality to be used in fixed points.\n    We do not use graph equality but instead type\n    equality.\n    '
    for (n, m) in zip(g1.nodes, g2.nodes):
        if n.type != m.type:
            return False
    return True