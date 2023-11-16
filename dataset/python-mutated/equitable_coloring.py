"""
Equitable coloring of graphs with bounded degree.
"""
from collections import defaultdict
import networkx as nx
__all__ = ['equitable_color']

@nx._dispatch
def is_coloring(G, coloring):
    if False:
        return 10
    'Determine if the coloring is a valid coloring for the graph G.'
    return all((coloring[s] != coloring[d] for (s, d) in G.edges))

@nx._dispatch
def is_equitable(G, coloring, num_colors=None):
    if False:
        print('Hello World!')
    'Determines if the coloring is valid and equitable for the graph G.'
    if not is_coloring(G, coloring):
        return False
    color_set_size = defaultdict(int)
    for color in coloring.values():
        color_set_size[color] += 1
    if num_colors is not None:
        for color in range(num_colors):
            if color not in color_set_size:
                color_set_size[color] = 0
    all_set_sizes = set(color_set_size.values())
    if len(all_set_sizes) == 0 and num_colors is None:
        return True
    elif len(all_set_sizes) == 1:
        return True
    elif len(all_set_sizes) == 2:
        (a, b) = list(all_set_sizes)
        return abs(a - b) <= 1
    else:
        return False

def make_C_from_F(F):
    if False:
        return 10
    C = defaultdict(list)
    for (node, color) in F.items():
        C[color].append(node)
    return C

def make_N_from_L_C(L, C):
    if False:
        return 10
    nodes = L.keys()
    colors = C.keys()
    return {(node, color): sum((1 for v in L[node] if v in C[color])) for node in nodes for color in colors}

def make_H_from_C_N(C, N):
    if False:
        i = 10
        return i + 15
    return {(c1, c2): sum((1 for node in C[c1] if N[node, c2] == 0)) for c1 in C for c2 in C}

def change_color(u, X, Y, N, H, F, C, L):
    if False:
        while True:
            i = 10
    "Change the color of 'u' from X to Y and update N, H, F, C."
    assert F[u] == X and X != Y
    F[u] = Y
    for k in C:
        if N[u, k] == 0:
            H[X, k] -= 1
            H[Y, k] += 1
    for v in L[u]:
        N[v, X] -= 1
        N[v, Y] += 1
        if N[v, X] == 0:
            H[F[v], X] += 1
        if N[v, Y] == 1:
            H[F[v], Y] -= 1
    C[X].remove(u)
    C[Y].append(u)

def move_witnesses(src_color, dst_color, N, H, F, C, T_cal, L):
    if False:
        print('Hello World!')
    'Move witness along a path from src_color to dst_color.'
    X = src_color
    while X != dst_color:
        Y = T_cal[X]
        w = next((x for x in C[X] if N[x, Y] == 0))
        change_color(w, X, Y, N=N, H=H, F=F, C=C, L=L)
        X = Y

@nx._dispatch
def pad_graph(G, num_colors):
    if False:
        i = 10
        return i + 15
    "Add a disconnected complete clique K_p such that the number of nodes in\n    the graph becomes a multiple of `num_colors`.\n\n    Assumes that the graph's nodes are labelled using integers.\n\n    Returns the number of nodes with each color.\n    "
    n_ = len(G)
    r = num_colors - 1
    s = n_ // (r + 1)
    if n_ != s * (r + 1):
        p = r + 1 - n_ % (r + 1)
        s += 1
        K = nx.relabel_nodes(nx.complete_graph(p), {idx: idx + n_ for idx in range(p)})
        G.add_edges_from(K.edges)
    return s

def procedure_P(V_minus, V_plus, N, H, F, C, L, excluded_colors=None):
    if False:
        return 10
    'Procedure P as described in the paper.'
    if excluded_colors is None:
        excluded_colors = set()
    A_cal = set()
    T_cal = {}
    R_cal = []
    reachable = [V_minus]
    marked = set(reachable)
    idx = 0
    while idx < len(reachable):
        pop = reachable[idx]
        idx += 1
        A_cal.add(pop)
        R_cal.append(pop)
        next_layer = []
        for k in C:
            if H[k, pop] > 0 and k not in A_cal and (k not in excluded_colors) and (k not in marked):
                next_layer.append(k)
        for dst in next_layer:
            T_cal[dst] = pop
        marked.update(next_layer)
        reachable.extend(next_layer)
    b = len(C) - len(A_cal)
    if V_plus in A_cal:
        move_witnesses(V_plus, V_minus, N=N, H=H, F=F, C=C, T_cal=T_cal, L=L)
    else:
        A_0 = set()
        A_cal_0 = set()
        num_terminal_sets_found = 0
        made_equitable = False
        for W_1 in R_cal[::-1]:
            for v in C[W_1]:
                X = None
                for U in C:
                    if N[v, U] == 0 and U in A_cal and (U != W_1):
                        X = U
                if X is None:
                    continue
                for U in C:
                    if N[v, U] >= 1 and U not in A_cal:
                        X_prime = U
                        w = v
                        try:
                            y = next((node for node in L[w] if F[node] == X_prime and N[node, W_1] == 1))
                        except StopIteration:
                            pass
                        else:
                            W = W_1
                            change_color(w, W, X, N=N, H=H, F=F, C=C, L=L)
                            move_witnesses(src_color=X, dst_color=V_minus, N=N, H=H, F=F, C=C, T_cal=T_cal, L=L)
                            change_color(y, X_prime, W, N=N, H=H, F=F, C=C, L=L)
                            procedure_P(V_minus=X_prime, V_plus=V_plus, N=N, H=H, C=C, F=F, L=L, excluded_colors=excluded_colors.union(A_cal))
                            made_equitable = True
                            break
                if made_equitable:
                    break
            else:
                A_cal_0.add(W_1)
                A_0.update(C[W_1])
                num_terminal_sets_found += 1
            if num_terminal_sets_found == b:
                B_cal_prime = set()
                T_cal_prime = {}
                reachable = [V_plus]
                marked = set(reachable)
                idx = 0
                while idx < len(reachable):
                    pop = reachable[idx]
                    idx += 1
                    B_cal_prime.add(pop)
                    next_layer = [k for k in C if H[pop, k] > 0 and k not in B_cal_prime and (k not in marked)]
                    for dst in next_layer:
                        T_cal_prime[pop] = dst
                    marked.update(next_layer)
                    reachable.extend(next_layer)
                I_set = set()
                I_covered = set()
                W_covering = {}
                B_prime = [node for k in B_cal_prime for node in C[k]]
                for z in C[V_plus] + B_prime:
                    if z in I_covered or F[z] not in B_cal_prime:
                        continue
                    I_set.add(z)
                    I_covered.add(z)
                    I_covered.update(list(L[z]))
                    for w in L[z]:
                        if F[w] in A_cal_0 and N[z, F[w]] == 1:
                            if w not in W_covering:
                                W_covering[w] = z
                            else:
                                z_1 = W_covering[w]
                                Z = F[z_1]
                                W = F[w]
                                move_witnesses(W, V_minus, N=N, H=H, F=F, C=C, T_cal=T_cal, L=L)
                                move_witnesses(V_plus, Z, N=N, H=H, F=F, C=C, T_cal=T_cal_prime, L=L)
                                change_color(z_1, Z, W, N=N, H=H, F=F, C=C, L=L)
                                W_plus = next((k for k in C if N[w, k] == 0 and k not in A_cal))
                                change_color(w, W, W_plus, N=N, H=H, F=F, C=C, L=L)
                                excluded_colors.update([k for k in C if k != W and k not in B_cal_prime])
                                procedure_P(V_minus=W, V_plus=W_plus, N=N, H=H, C=C, F=F, L=L, excluded_colors=excluded_colors)
                                made_equitable = True
                                break
                    if made_equitable:
                        break
                else:
                    assert False, 'Must find a w which is the solo neighbor of two vertices in B_cal_prime.'
            if made_equitable:
                break

@nx._dispatch
def equitable_color(G, num_colors):
    if False:
        i = 10
        return i + 15
    'Provides an equitable coloring for nodes of `G`.\n\n    Attempts to color a graph using `num_colors` colors, where no neighbors of\n    a node can have same color as the node itself and the number of nodes with\n    each color differ by at most 1. `num_colors` must be greater than the\n    maximum degree of `G`. The algorithm is described in [1]_ and has\n    complexity O(num_colors * n**2).\n\n    Parameters\n    ----------\n    G : networkX graph\n       The nodes of this graph will be colored.\n\n    num_colors : number of colors to use\n       This number must be at least one more than the maximum degree of nodes\n       in the graph.\n\n    Returns\n    -------\n    A dictionary with keys representing nodes and values representing\n    corresponding coloring.\n\n    Examples\n    --------\n    >>> G = nx.cycle_graph(4)\n    >>> nx.coloring.equitable_color(G, num_colors=3)  # doctest: +SKIP\n    {0: 2, 1: 1, 2: 2, 3: 0}\n\n    Raises\n    ------\n    NetworkXAlgorithmError\n        If `num_colors` is not at least the maximum degree of the graph `G`\n\n    References\n    ----------\n    .. [1] Kierstead, H. A., Kostochka, A. V., Mydlarz, M., & Szemerédi, E.\n        (2010). A fast algorithm for equitable coloring. Combinatorica, 30(2),\n        217-224.\n    '
    nodes_to_int = {}
    int_to_nodes = {}
    for (idx, node) in enumerate(G.nodes):
        nodes_to_int[node] = idx
        int_to_nodes[idx] = node
    G = nx.relabel_nodes(G, nodes_to_int, copy=True)
    if len(G.nodes) > 0:
        r_ = max((G.degree(node) for node in G.nodes))
    else:
        r_ = 0
    if r_ >= num_colors:
        raise nx.NetworkXAlgorithmError(f'Graph has maximum degree {r_}, needs {r_ + 1} (> {num_colors}) colors for guaranteed coloring.')
    pad_graph(G, num_colors)
    L_ = {node: [] for node in G.nodes}
    F = {node: idx % num_colors for (idx, node) in enumerate(G.nodes)}
    C = make_C_from_F(F)
    N = make_N_from_L_C(L_, C)
    H = make_H_from_C_N(C, N)
    edges_seen = set()
    for u in sorted(G.nodes):
        for v in sorted(G.neighbors(u)):
            if (v, u) in edges_seen:
                continue
            edges_seen.add((u, v))
            L_[u].append(v)
            L_[v].append(u)
            N[u, F[v]] += 1
            N[v, F[u]] += 1
            if F[u] != F[v]:
                if N[u, F[v]] == 1:
                    H[F[u], F[v]] -= 1
                if N[v, F[u]] == 1:
                    H[F[v], F[u]] -= 1
        if N[u, F[u]] != 0:
            Y = next((k for k in C if N[u, k] == 0))
            X = F[u]
            change_color(u, X, Y, N=N, H=H, F=F, C=C, L=L_)
            procedure_P(V_minus=X, V_plus=Y, N=N, H=H, F=F, C=C, L=L_)
    return {int_to_nodes[x]: F[x] for x in int_to_nodes}