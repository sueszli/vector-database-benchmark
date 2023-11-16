"""Generates graphs resembling the Internet Autonomous System network"""
import networkx as nx
from networkx.utils import py_random_state
__all__ = ['random_internet_as_graph']

def uniform_int_from_avg(a, m, seed):
    if False:
        print('Hello World!')
    "Pick a random integer with uniform probability.\n\n    Returns a random integer uniformly taken from a distribution with\n    minimum value 'a' and average value 'm', X~U(a,b), E[X]=m, X in N where\n    b = 2*m - a.\n\n    Notes\n    -----\n    p = (b-floor(b))/2\n    X = X1 + X2; X1~U(a,floor(b)), X2~B(p)\n    E[X] = E[X1] + E[X2] = (floor(b)+a)/2 + (b-floor(b))/2 = (b+a)/2 = m\n    "
    from math import floor
    assert m >= a
    b = 2 * m - a
    p = (b - floor(b)) / 2
    X1 = round(seed.random() * (floor(b) - a) + a)
    if seed.random() < p:
        X2 = 1
    else:
        X2 = 0
    return X1 + X2

def choose_pref_attach(degs, seed):
    if False:
        while True:
            i = 10
    'Pick a random value, with a probability given by its weight.\n\n    Returns a random choice among degs keys, each of which has a\n    probability proportional to the corresponding dictionary value.\n\n    Parameters\n    ----------\n    degs: dictionary\n        It contains the possible values (keys) and the corresponding\n        probabilities (values)\n    seed: random state\n\n    Returns\n    -------\n    v: object\n        A key of degs or None if degs is empty\n    '
    if len(degs) == 0:
        return None
    s = sum(degs.values())
    if s == 0:
        return seed.choice(list(degs.keys()))
    v = seed.random() * s
    nodes = list(degs.keys())
    i = 0
    acc = degs[nodes[i]]
    while v > acc:
        i += 1
        acc += degs[nodes[i]]
    return nodes[i]

class AS_graph_generator:
    """Generates random internet AS graphs."""

    def __init__(self, n, seed):
        if False:
            for i in range(10):
                print('nop')
        'Initializes variables. Immediate numbers are taken from [1].\n\n        Parameters\n        ----------\n        n: integer\n            Number of graph nodes\n        seed: random state\n            Indicator of random number generation state.\n            See :ref:`Randomness<randomness>`.\n\n        Returns\n        -------\n        GG: AS_graph_generator object\n\n        References\n        ----------\n        [1] A. Elmokashfi, A. Kvalbein and C. Dovrolis, "On the Scalability of\n        BGP: The Role of Topology Growth," in IEEE Journal on Selected Areas\n        in Communications, vol. 28, no. 8, pp. 1250-1261, October 2010.\n        '
        self.seed = seed
        self.n_t = min(n, round(self.seed.random() * 2 + 4))
        self.n_m = round(0.15 * n)
        self.n_cp = round(0.05 * n)
        self.n_c = max(0, n - self.n_t - self.n_m - self.n_cp)
        self.d_m = 2 + 2.5 * n / 10000
        self.d_cp = 2 + 1.5 * n / 10000
        self.d_c = 1 + 5 * n / 100000
        self.p_m_m = 1 + 2 * n / 10000
        self.p_cp_m = 0.2 + 2 * n / 10000
        self.p_cp_cp = 0.05 + 2 * n / 100000
        self.t_m = 0.375
        self.t_cp = 0.375
        self.t_c = 0.125

    def t_graph(self):
        if False:
            print('Hello World!')
        'Generates the core mesh network of tier one nodes of a AS graph.\n\n        Returns\n        -------\n        G: Networkx Graph\n            Core network\n        '
        self.G = nx.Graph()
        for i in range(self.n_t):
            self.G.add_node(i, type='T')
            for r in self.regions:
                self.regions[r].add(i)
            for j in self.G.nodes():
                if i != j:
                    self.add_edge(i, j, 'peer')
            self.customers[i] = set()
            self.providers[i] = set()
        return self.G

    def add_edge(self, i, j, kind):
        if False:
            return 10
        if kind == 'transit':
            customer = str(i)
        else:
            customer = 'none'
        self.G.add_edge(i, j, type=kind, customer=customer)

    def choose_peer_pref_attach(self, node_list):
        if False:
            print('Hello World!')
        'Pick a node with a probability weighted by its peer degree.\n\n        Pick a node from node_list with preferential attachment\n        computed only on their peer degree\n        '
        d = {}
        for n in node_list:
            d[n] = self.G.nodes[n]['peers']
        return choose_pref_attach(d, self.seed)

    def choose_node_pref_attach(self, node_list):
        if False:
            i = 10
            return i + 15
        'Pick a node with a probability weighted by its degree.\n\n        Pick a node from node_list with preferential attachment\n        computed on their degree\n        '
        degs = dict(self.G.degree(node_list))
        return choose_pref_attach(degs, self.seed)

    def add_customer(self, i, j):
        if False:
            while True:
                i = 10
        "Keep the dictionaries 'customers' and 'providers' consistent."
        self.customers[j].add(i)
        self.providers[i].add(j)
        for z in self.providers[j]:
            self.customers[z].add(i)
            self.providers[i].add(z)

    def add_node(self, i, kind, reg2prob, avg_deg, t_edge_prob):
        if False:
            return 10
        "Add a node and its customer transit edges to the graph.\n\n        Parameters\n        ----------\n        i: object\n            Identifier of the new node\n        kind: string\n            Type of the new node. Options are: 'M' for middle node, 'CP' for\n            content provider and 'C' for customer.\n        reg2prob: float\n            Probability the new node can be in two different regions.\n        avg_deg: float\n            Average number of transit nodes of which node i is customer.\n        t_edge_prob: float\n            Probability node i establish a customer transit edge with a tier\n            one (T) node\n\n        Returns\n        -------\n        i: object\n            Identifier of the new node\n        "
        regs = 1
        if self.seed.random() < reg2prob:
            regs = 2
        node_options = set()
        self.G.add_node(i, type=kind, peers=0)
        self.customers[i] = set()
        self.providers[i] = set()
        self.nodes[kind].add(i)
        for r in self.seed.sample(list(self.regions), regs):
            node_options = node_options.union(self.regions[r])
            self.regions[r].add(i)
        edge_num = uniform_int_from_avg(1, avg_deg, self.seed)
        t_options = node_options.intersection(self.nodes['T'])
        m_options = node_options.intersection(self.nodes['M'])
        if i in m_options:
            m_options.remove(i)
        d = 0
        while d < edge_num and (len(t_options) > 0 or len(m_options) > 0):
            if len(m_options) == 0 or (len(t_options) > 0 and self.seed.random() < t_edge_prob):
                j = self.choose_node_pref_attach(t_options)
                t_options.remove(j)
            else:
                j = self.choose_node_pref_attach(m_options)
                m_options.remove(j)
            self.add_edge(i, j, 'transit')
            self.add_customer(i, j)
            d += 1
        return i

    def add_m_peering_link(self, m, to_kind):
        if False:
            print('Hello World!')
        'Add a peering link between two middle tier (M) nodes.\n\n        Target node j is drawn considering a preferential attachment based on\n        other M node peering degree.\n\n        Parameters\n        ----------\n        m: object\n            Node identifier\n        to_kind: string\n            type for target node j (must be always M)\n\n        Returns\n        -------\n        success: boolean\n        '
        node_options = self.nodes['M'].difference(self.customers[m])
        node_options = node_options.difference(self.providers[m])
        if m in node_options:
            node_options.remove(m)
        for j in self.G.neighbors(m):
            if j in node_options:
                node_options.remove(j)
        if len(node_options) > 0:
            j = self.choose_peer_pref_attach(node_options)
            self.add_edge(m, j, 'peer')
            self.G.nodes[m]['peers'] += 1
            self.G.nodes[j]['peers'] += 1
            return True
        else:
            return False

    def add_cp_peering_link(self, cp, to_kind):
        if False:
            i = 10
            return i + 15
        'Add a peering link to a content provider (CP) node.\n\n        Target node j can be CP or M and it is drawn uniformly among the nodes\n        belonging to the same region as cp.\n\n        Parameters\n        ----------\n        cp: object\n            Node identifier\n        to_kind: string\n            type for target node j (must be M or CP)\n\n        Returns\n        -------\n        success: boolean\n        '
        node_options = set()
        for r in self.regions:
            if cp in self.regions[r]:
                node_options = node_options.union(self.regions[r])
        node_options = self.nodes[to_kind].intersection(node_options)
        if cp in node_options:
            node_options.remove(cp)
        node_options = node_options.difference(self.providers[cp])
        for j in self.G.neighbors(cp):
            if j in node_options:
                node_options.remove(j)
        if len(node_options) > 0:
            j = self.seed.sample(list(node_options), 1)[0]
            self.add_edge(cp, j, 'peer')
            self.G.nodes[cp]['peers'] += 1
            self.G.nodes[j]['peers'] += 1
            return True
        else:
            return False

    def graph_regions(self, rn):
        if False:
            i = 10
            return i + 15
        'Initializes AS network regions.\n\n        Parameters\n        ----------\n        rn: integer\n            Number of regions\n        '
        self.regions = {}
        for i in range(rn):
            self.regions['REG' + str(i)] = set()

    def add_peering_links(self, from_kind, to_kind):
        if False:
            return 10
        'Utility function to add peering links among node groups.'
        peer_link_method = None
        if from_kind == 'M':
            peer_link_method = self.add_m_peering_link
            m = self.p_m_m
        if from_kind == 'CP':
            peer_link_method = self.add_cp_peering_link
            if to_kind == 'M':
                m = self.p_cp_m
            else:
                m = self.p_cp_cp
        for i in self.nodes[from_kind]:
            num = uniform_int_from_avg(0, m, self.seed)
            for _ in range(num):
                peer_link_method(i, to_kind)

    def generate(self):
        if False:
            print('Hello World!')
        'Generates a random AS network graph as described in [1].\n\n        Returns\n        -------\n        G: Graph object\n\n        Notes\n        -----\n        The process steps are the following: first we create the core network\n        of tier one nodes, then we add the middle tier (M), the content\n        provider (CP) and the customer (C) nodes along with their transit edges\n        (link i,j means i is customer of j). Finally we add peering links\n        between M nodes, between M and CP nodes and between CP node couples.\n        For a detailed description of the algorithm, please refer to [1].\n\n        References\n        ----------\n        [1] A. Elmokashfi, A. Kvalbein and C. Dovrolis, "On the Scalability of\n        BGP: The Role of Topology Growth," in IEEE Journal on Selected Areas\n        in Communications, vol. 28, no. 8, pp. 1250-1261, October 2010.\n        '
        self.graph_regions(5)
        self.customers = {}
        self.providers = {}
        self.nodes = {'T': set(), 'M': set(), 'CP': set(), 'C': set()}
        self.t_graph()
        self.nodes['T'] = set(self.G.nodes())
        i = len(self.nodes['T'])
        for _ in range(self.n_m):
            self.nodes['M'].add(self.add_node(i, 'M', 0.2, self.d_m, self.t_m))
            i += 1
        for _ in range(self.n_cp):
            self.nodes['CP'].add(self.add_node(i, 'CP', 0.05, self.d_cp, self.t_cp))
            i += 1
        for _ in range(self.n_c):
            self.nodes['C'].add(self.add_node(i, 'C', 0, self.d_c, self.t_c))
            i += 1
        self.add_peering_links('M', 'M')
        self.add_peering_links('CP', 'M')
        self.add_peering_links('CP', 'CP')
        return self.G

@py_random_state(1)
@nx._dispatch(graphs=None)
def random_internet_as_graph(n, seed=None):
    if False:
        return 10
    'Generates a random undirected graph resembling the Internet AS network\n\n    Parameters\n    ----------\n    n: integer in [1000, 10000]\n        Number of graph nodes\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    G: Networkx Graph object\n        A randomly generated undirected graph\n\n    Notes\n    -----\n    This algorithm returns an undirected graph resembling the Internet\n    Autonomous System (AS) network, it uses the approach by Elmokashfi et al.\n    [1]_ and it grants the properties described in the related paper [1]_.\n\n    Each node models an autonomous system, with an attribute \'type\' specifying\n    its kind; tier-1 (T), mid-level (M), customer (C) or content-provider (CP).\n    Each edge models an ADV communication link (hence, bidirectional) with\n    attributes:\n\n      - type: transit|peer, the kind of commercial agreement between nodes;\n      - customer: <node id>, the identifier of the node acting as customer\n        (\'none\' if type is peer).\n\n    References\n    ----------\n    .. [1] A. Elmokashfi, A. Kvalbein and C. Dovrolis, "On the Scalability of\n       BGP: The Role of Topology Growth," in IEEE Journal on Selected Areas\n       in Communications, vol. 28, no. 8, pp. 1250-1261, October 2010.\n    '
    GG = AS_graph_generator(n, seed)
    G = GG.generate()
    return G