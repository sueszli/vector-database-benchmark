from sympy.ntheory.primetest import isprime
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.printing.defaults import DefaultPrinting
from sympy.combinatorics.free_groups import free_group

class PolycyclicGroup(DefaultPrinting):
    is_group = True
    is_solvable = True

    def __init__(self, pc_sequence, pc_series, relative_order, collector=None):
        if False:
            i = 10
            return i + 15
        '\n\n        Parameters\n        ==========\n\n        pc_sequence : list\n            A sequence of elements whose classes generate the cyclic factor\n            groups of pc_series.\n        pc_series : list\n            A subnormal sequence of subgroups where each factor group is cyclic.\n        relative_order : list\n            The orders of factor groups of pc_series.\n        collector : Collector\n            By default, it is None. Collector class provides the\n            polycyclic presentation with various other functionalities.\n\n        '
        self.pcgs = pc_sequence
        self.pc_series = pc_series
        self.relative_order = relative_order
        self.collector = Collector(self.pcgs, pc_series, relative_order) if not collector else collector

    def is_prime_order(self):
        if False:
            return 10
        return all((isprime(order) for order in self.relative_order))

    def length(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.pcgs)

class Collector(DefaultPrinting):
    """
    References
    ==========

    .. [1] Holt, D., Eick, B., O'Brien, E.
           "Handbook of Computational Group Theory"
           Section 8.1.3
    """

    def __init__(self, pcgs, pc_series, relative_order, free_group_=None, pc_presentation=None):
        if False:
            while True:
                i = 10
        '\n\n        Most of the parameters for the Collector class are the same as for PolycyclicGroup.\n        Others are described below.\n\n        Parameters\n        ==========\n\n        free_group_ : tuple\n            free_group_ provides the mapping of polycyclic generating\n            sequence with the free group elements.\n        pc_presentation : dict\n            Provides the presentation of polycyclic groups with the\n            help of power and conjugate relators.\n\n        See Also\n        ========\n\n        PolycyclicGroup\n\n        '
        self.pcgs = pcgs
        self.pc_series = pc_series
        self.relative_order = relative_order
        self.free_group = free_group('x:{}'.format(len(pcgs)))[0] if not free_group_ else free_group_
        self.index = {s: i for (i, s) in enumerate(self.free_group.symbols)}
        self.pc_presentation = self.pc_relators()

    def minimal_uncollected_subword(self, word):
        if False:
            print('Hello World!')
        '\n        Returns the minimal uncollected subwords.\n\n        Explanation\n        ===========\n\n        A word ``v`` defined on generators in ``X`` is a minimal\n        uncollected subword of the word ``w`` if ``v`` is a subword\n        of ``w`` and it has one of the following form\n\n        * `v = {x_{i+1}}^{a_j}x_i`\n\n        * `v = {x_{i+1}}^{a_j}{x_i}^{-1}`\n\n        * `v = {x_i}^{a_j}`\n\n        for `a_j` not in `\\{1, \\ldots, s-1\\}`. Where, ``s`` is the power\n        exponent of the corresponding generator.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.named_groups import SymmetricGroup\n        >>> from sympy.combinatorics import free_group\n        >>> G = SymmetricGroup(4)\n        >>> PcGroup = G.polycyclic_group()\n        >>> collector = PcGroup.collector\n        >>> F, x1, x2 = free_group("x1, x2")\n        >>> word = x2**2*x1**7\n        >>> collector.minimal_uncollected_subword(word)\n        ((x2, 2),)\n\n        '
        if not word:
            return None
        array = word.array_form
        re = self.relative_order
        index = self.index
        for i in range(len(array)):
            (s1, e1) = array[i]
            if re[index[s1]] and (e1 < 0 or e1 > re[index[s1]] - 1):
                return ((s1, e1),)
        for i in range(len(array) - 1):
            (s1, e1) = array[i]
            (s2, e2) = array[i + 1]
            if index[s1] > index[s2]:
                e = 1 if e2 > 0 else -1
                return ((s1, e1), (s2, e))
        return None

    def relations(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Separates the given relators of pc presentation in power and\n        conjugate relations.\n\n        Returns\n        =======\n\n        (power_rel, conj_rel)\n            Separates pc presentation into power and conjugate relations.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.named_groups import SymmetricGroup\n        >>> G = SymmetricGroup(3)\n        >>> PcGroup = G.polycyclic_group()\n        >>> collector = PcGroup.collector\n        >>> power_rel, conj_rel = collector.relations()\n        >>> power_rel\n        {x0**2: (), x1**3: ()}\n        >>> conj_rel\n        {x0**-1*x1*x0: x1**2}\n\n        See Also\n        ========\n\n        pc_relators\n\n        '
        power_relators = {}
        conjugate_relators = {}
        for (key, value) in self.pc_presentation.items():
            if len(key.array_form) == 1:
                power_relators[key] = value
            else:
                conjugate_relators[key] = value
        return (power_relators, conjugate_relators)

    def subword_index(self, word, w):
        if False:
            i = 10
            return i + 15
        '\n        Returns the start and ending index of a given\n        subword in a word.\n\n        Parameters\n        ==========\n\n        word : FreeGroupElement\n            word defined on free group elements for a\n            polycyclic group.\n        w : FreeGroupElement\n            subword of a given word, whose starting and\n            ending index to be computed.\n\n        Returns\n        =======\n\n        (i, j)\n            A tuple containing starting and ending index of ``w``\n            in the given word.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.named_groups import SymmetricGroup\n        >>> from sympy.combinatorics import free_group\n        >>> G = SymmetricGroup(4)\n        >>> PcGroup = G.polycyclic_group()\n        >>> collector = PcGroup.collector\n        >>> F, x1, x2 = free_group("x1, x2")\n        >>> word = x2**2*x1**7\n        >>> w = x2**2*x1\n        >>> collector.subword_index(word, w)\n        (0, 3)\n        >>> w = x1**7\n        >>> collector.subword_index(word, w)\n        (2, 9)\n\n        '
        low = -1
        high = -1
        for i in range(len(word) - len(w) + 1):
            if word.subword(i, i + len(w)) == w:
                low = i
                high = i + len(w)
                break
        if low == high == -1:
            return (-1, -1)
        return (low, high)

    def map_relation(self, w):
        if False:
            while True:
                i = 10
        '\n        Return a conjugate relation.\n\n        Explanation\n        ===========\n\n        Given a word formed by two free group elements, the\n        corresponding conjugate relation with those free\n        group elements is formed and mapped with the collected\n        word in the polycyclic presentation.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.named_groups import SymmetricGroup\n        >>> from sympy.combinatorics import free_group\n        >>> G = SymmetricGroup(3)\n        >>> PcGroup = G.polycyclic_group()\n        >>> collector = PcGroup.collector\n        >>> F, x0, x1 = free_group("x0, x1")\n        >>> w = x1*x0\n        >>> collector.map_relation(w)\n        x1**2\n\n        See Also\n        ========\n\n        pc_presentation\n\n        '
        array = w.array_form
        s1 = array[0][0]
        s2 = array[1][0]
        key = ((s2, -1), (s1, 1), (s2, 1))
        key = self.free_group.dtype(key)
        return self.pc_presentation[key]

    def collected_word(self, word):
        if False:
            while True:
                i = 10
        '\n        Return the collected form of a word.\n\n        Explanation\n        ===========\n\n        A word ``w`` is called collected, if `w = {x_{i_1}}^{a_1} * \\ldots *\n        {x_{i_r}}^{a_r}` with `i_1 < i_2< \\ldots < i_r` and `a_j` is in\n        `\\{1, \\ldots, {s_j}-1\\}`.\n\n        Otherwise w is uncollected.\n\n        Parameters\n        ==========\n\n        word : FreeGroupElement\n            An uncollected word.\n\n        Returns\n        =======\n\n        word\n            A collected word of form `w = {x_{i_1}}^{a_1}, \\ldots,\n            {x_{i_r}}^{a_r}` with `i_1, i_2, \\ldots, i_r` and `a_j \\in\n            \\{1, \\ldots, {s_j}-1\\}`.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.named_groups import SymmetricGroup\n        >>> from sympy.combinatorics.perm_groups import PermutationGroup\n        >>> from sympy.combinatorics import free_group\n        >>> G = SymmetricGroup(4)\n        >>> PcGroup = G.polycyclic_group()\n        >>> collector = PcGroup.collector\n        >>> F, x0, x1, x2, x3 = free_group("x0, x1, x2, x3")\n        >>> word = x3*x2*x1*x0\n        >>> collected_word = collector.collected_word(word)\n        >>> free_to_perm = {}\n        >>> free_group = collector.free_group\n        >>> for sym, gen in zip(free_group.symbols, collector.pcgs):\n        ...     free_to_perm[sym] = gen\n        >>> G1 = PermutationGroup()\n        >>> for w in word:\n        ...     sym = w[0]\n        ...     perm = free_to_perm[sym]\n        ...     G1 = PermutationGroup([perm] + G1.generators)\n        >>> G2 = PermutationGroup()\n        >>> for w in collected_word:\n        ...     sym = w[0]\n        ...     perm = free_to_perm[sym]\n        ...     G2 = PermutationGroup([perm] + G2.generators)\n\n        The two are not identical, but they are equivalent:\n\n        >>> G1.equals(G2), G1 == G2\n        (True, False)\n\n        See Also\n        ========\n\n        minimal_uncollected_subword\n\n        '
        free_group = self.free_group
        while True:
            w = self.minimal_uncollected_subword(word)
            if not w:
                break
            (low, high) = self.subword_index(word, free_group.dtype(w))
            if low == -1:
                continue
            (s1, e1) = w[0]
            if len(w) == 1:
                re = self.relative_order[self.index[s1]]
                q = e1 // re
                r = e1 - q * re
                key = ((w[0][0], re),)
                key = free_group.dtype(key)
                if self.pc_presentation[key]:
                    presentation = self.pc_presentation[key].array_form
                    (sym, exp) = presentation[0]
                    word_ = ((w[0][0], r), (sym, q * exp))
                    word_ = free_group.dtype(word_)
                elif r != 0:
                    word_ = ((w[0][0], r),)
                    word_ = free_group.dtype(word_)
                else:
                    word_ = None
                word = word.eliminate_word(free_group.dtype(w), word_)
            if len(w) == 2 and w[1][1] > 0:
                (s2, e2) = w[1]
                s2 = ((s2, 1),)
                s2 = free_group.dtype(s2)
                word_ = self.map_relation(free_group.dtype(w))
                word_ = s2 * word_ ** e1
                word_ = free_group.dtype(word_)
                word = word.substituted_word(low, high, word_)
            elif len(w) == 2 and w[1][1] < 0:
                (s2, e2) = w[1]
                s2 = ((s2, 1),)
                s2 = free_group.dtype(s2)
                word_ = self.map_relation(free_group.dtype(w))
                word_ = s2 ** (-1) * word_ ** e1
                word_ = free_group.dtype(word_)
                word = word.substituted_word(low, high, word_)
        return word

    def pc_relators(self):
        if False:
            while True:
                i = 10
        '\n        Return the polycyclic presentation.\n\n        Explanation\n        ===========\n\n        There are two types of relations used in polycyclic\n        presentation.\n\n        * Power relations : Power relators are of the form `x_i^{re_i}`,\n          where `i \\in \\{0, \\ldots, \\mathrm{len(pcgs)}\\}`, ``x`` represents polycyclic\n          generator and ``re`` is the corresponding relative order.\n\n        * Conjugate relations : Conjugate relators are of the form `x_j^-1x_ix_j`,\n          where `j < i \\in \\{0, \\ldots, \\mathrm{len(pcgs)}\\}`.\n\n        Returns\n        =======\n\n        A dictionary with power and conjugate relations as key and\n        their collected form as corresponding values.\n\n        Notes\n        =====\n\n        Identity Permutation is mapped with empty ``()``.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.named_groups import SymmetricGroup\n        >>> from sympy.combinatorics.permutations import Permutation\n        >>> S = SymmetricGroup(49).sylow_subgroup(7)\n        >>> der = S.derived_series()\n        >>> G = der[len(der)-2]\n        >>> PcGroup = G.polycyclic_group()\n        >>> collector = PcGroup.collector\n        >>> pcgs = PcGroup.pcgs\n        >>> len(pcgs)\n        6\n        >>> free_group = collector.free_group\n        >>> pc_resentation = collector.pc_presentation\n        >>> free_to_perm = {}\n        >>> for s, g in zip(free_group.symbols, pcgs):\n        ...     free_to_perm[s] = g\n\n        >>> for k, v in pc_resentation.items():\n        ...     k_array = k.array_form\n        ...     if v != ():\n        ...        v_array = v.array_form\n        ...     lhs = Permutation()\n        ...     for gen in k_array:\n        ...         s = gen[0]\n        ...         e = gen[1]\n        ...         lhs = lhs*free_to_perm[s]**e\n        ...     if v == ():\n        ...         assert lhs.is_identity\n        ...         continue\n        ...     rhs = Permutation()\n        ...     for gen in v_array:\n        ...         s = gen[0]\n        ...         e = gen[1]\n        ...         rhs = rhs*free_to_perm[s]**e\n        ...     assert lhs == rhs\n\n        '
        free_group = self.free_group
        rel_order = self.relative_order
        pc_relators = {}
        perm_to_free = {}
        pcgs = self.pcgs
        for (gen, s) in zip(pcgs, free_group.generators):
            perm_to_free[gen ** (-1)] = s ** (-1)
            perm_to_free[gen] = s
        pcgs = pcgs[::-1]
        series = self.pc_series[::-1]
        rel_order = rel_order[::-1]
        collected_gens = []
        for (i, gen) in enumerate(pcgs):
            re = rel_order[i]
            relation = perm_to_free[gen] ** re
            G = series[i]
            l = G.generator_product(gen ** re, original=True)
            l.reverse()
            word = free_group.identity
            for g in l:
                word = word * perm_to_free[g]
            word = self.collected_word(word)
            pc_relators[relation] = word if word else ()
            self.pc_presentation = pc_relators
            collected_gens.append(gen)
            if len(collected_gens) > 1:
                conj = collected_gens[len(collected_gens) - 1]
                conjugator = perm_to_free[conj]
                for j in range(len(collected_gens) - 1):
                    conjugated = perm_to_free[collected_gens[j]]
                    relation = conjugator ** (-1) * conjugated * conjugator
                    gens = conj ** (-1) * collected_gens[j] * conj
                    l = G.generator_product(gens, original=True)
                    l.reverse()
                    word = free_group.identity
                    for g in l:
                        word = word * perm_to_free[g]
                    word = self.collected_word(word)
                    pc_relators[relation] = word if word else ()
                    self.pc_presentation = pc_relators
        return pc_relators

    def exponent_vector(self, element):
        if False:
            print('Hello World!')
        '\n        Return the exponent vector of length equal to the\n        length of polycyclic generating sequence.\n\n        Explanation\n        ===========\n\n        For a given generator/element ``g`` of the polycyclic group,\n        it can be represented as `g = {x_1}^{e_1}, \\ldots, {x_n}^{e_n}`,\n        where `x_i` represents polycyclic generators and ``n`` is\n        the number of generators in the free_group equal to the length\n        of pcgs.\n\n        Parameters\n        ==========\n\n        element : Permutation\n            Generator of a polycyclic group.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.named_groups import SymmetricGroup\n        >>> from sympy.combinatorics.permutations import Permutation\n        >>> G = SymmetricGroup(4)\n        >>> PcGroup = G.polycyclic_group()\n        >>> collector = PcGroup.collector\n        >>> pcgs = PcGroup.pcgs\n        >>> collector.exponent_vector(G[0])\n        [1, 0, 0, 0]\n        >>> exp = collector.exponent_vector(G[1])\n        >>> g = Permutation()\n        >>> for i in range(len(exp)):\n        ...     g = g*pcgs[i]**exp[i] if exp[i] else g\n        >>> assert g == G[1]\n\n        References\n        ==========\n\n        .. [1] Holt, D., Eick, B., O\'Brien, E.\n               "Handbook of Computational Group Theory"\n               Section 8.1.1, Definition 8.4\n\n        '
        free_group = self.free_group
        G = PermutationGroup()
        for g in self.pcgs:
            G = PermutationGroup([g] + G.generators)
        gens = G.generator_product(element, original=True)
        gens.reverse()
        perm_to_free = {}
        for (sym, g) in zip(free_group.generators, self.pcgs):
            perm_to_free[g ** (-1)] = sym ** (-1)
            perm_to_free[g] = sym
        w = free_group.identity
        for g in gens:
            w = w * perm_to_free[g]
        word = self.collected_word(w)
        index = self.index
        exp_vector = [0] * len(free_group)
        word = word.array_form
        for t in word:
            exp_vector[index[t[0]]] = t[1]
        return exp_vector

    def depth(self, element):
        if False:
            print('Hello World!')
        '\n        Return the depth of a given element.\n\n        Explanation\n        ===========\n\n        The depth of a given element ``g`` is defined by\n        `\\mathrm{dep}[g] = i` if `e_1 = e_2 = \\ldots = e_{i-1} = 0`\n        and `e_i != 0`, where ``e`` represents the exponent-vector.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.named_groups import SymmetricGroup\n        >>> G = SymmetricGroup(3)\n        >>> PcGroup = G.polycyclic_group()\n        >>> collector = PcGroup.collector\n        >>> collector.depth(G[0])\n        2\n        >>> collector.depth(G[1])\n        1\n\n        References\n        ==========\n\n        .. [1] Holt, D., Eick, B., O\'Brien, E.\n               "Handbook of Computational Group Theory"\n               Section 8.1.1, Definition 8.5\n\n        '
        exp_vector = self.exponent_vector(element)
        return next((i + 1 for (i, x) in enumerate(exp_vector) if x), len(self.pcgs) + 1)

    def leading_exponent(self, element):
        if False:
            return 10
        '\n        Return the leading non-zero exponent.\n\n        Explanation\n        ===========\n\n        The leading exponent for a given element `g` is defined\n        by `\\mathrm{leading\\_exponent}[g]` `= e_i`, if `\\mathrm{depth}[g] = i`.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.named_groups import SymmetricGroup\n        >>> G = SymmetricGroup(3)\n        >>> PcGroup = G.polycyclic_group()\n        >>> collector = PcGroup.collector\n        >>> collector.leading_exponent(G[1])\n        1\n\n        '
        exp_vector = self.exponent_vector(element)
        depth = self.depth(element)
        if depth != len(self.pcgs) + 1:
            return exp_vector[depth - 1]
        return None

    def _sift(self, z, g):
        if False:
            print('Hello World!')
        h = g
        d = self.depth(h)
        while d < len(self.pcgs) and z[d - 1] != 1:
            k = z[d - 1]
            e = self.leading_exponent(h) * self.leading_exponent(k) ** (-1)
            e = e % self.relative_order[d - 1]
            h = k ** (-e) * h
            d = self.depth(h)
        return h

    def induced_pcgs(self, gens):
        if False:
            while True:
                i = 10
        '\n\n        Parameters\n        ==========\n\n        gens : list\n            A list of generators on which polycyclic subgroup\n            is to be defined.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.named_groups import SymmetricGroup\n        >>> S = SymmetricGroup(8)\n        >>> G = S.sylow_subgroup(2)\n        >>> PcGroup = G.polycyclic_group()\n        >>> collector = PcGroup.collector\n        >>> gens = [G[0], G[1]]\n        >>> ipcgs = collector.induced_pcgs(gens)\n        >>> [gen.order() for gen in ipcgs]\n        [2, 2, 2]\n        >>> G = S.sylow_subgroup(3)\n        >>> PcGroup = G.polycyclic_group()\n        >>> collector = PcGroup.collector\n        >>> gens = [G[0], G[1]]\n        >>> ipcgs = collector.induced_pcgs(gens)\n        >>> [gen.order() for gen in ipcgs]\n        [3]\n\n        '
        z = [1] * len(self.pcgs)
        G = gens
        while G:
            g = G.pop(0)
            h = self._sift(z, g)
            d = self.depth(h)
            if d < len(self.pcgs):
                for gen in z:
                    if gen != 1:
                        G.append(h ** (-1) * gen ** (-1) * h * gen)
                z[d - 1] = h
        z = [gen for gen in z if gen != 1]
        return z

    def constructive_membership_test(self, ipcgs, g):
        if False:
            print('Hello World!')
        '\n        Return the exponent vector for induced pcgs.\n        '
        e = [0] * len(ipcgs)
        h = g
        d = self.depth(h)
        for (i, gen) in enumerate(ipcgs):
            while self.depth(gen) == d:
                f = self.leading_exponent(h) * self.leading_exponent(gen)
                f = f % self.relative_order[d - 1]
                h = gen ** (-f) * h
                e[i] = f
                d = self.depth(h)
        if h == 1:
            return e
        return False