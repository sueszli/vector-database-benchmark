from sympy.combinatorics.free_groups import free_group
from sympy.printing.defaults import DefaultPrinting
from itertools import chain, product
from bisect import bisect_left

class CosetTable(DefaultPrinting):
    """

    Properties
    ==========

    [1] `0 \\in \\Omega` and `\\tau(1) = \\epsilon`
    [2] `\\alpha^x = \\beta \\Leftrightarrow \\beta^{x^{-1}} = \\alpha`
    [3] If `\\alpha^x = \\beta`, then `H \\tau(\\alpha)x = H \\tau(\\beta)`
    [4] `\\forall \\alpha \\in \\Omega, 1^{\\tau(\\alpha)} = \\alpha`

    References
    ==========

    .. [1] Holt, D., Eick, B., O'Brien, E.
           "Handbook of Computational Group Theory"

    .. [2] John J. Cannon; Lucien A. Dimino; George Havas; Jane M. Watson
           Mathematics of Computation, Vol. 27, No. 123. (Jul., 1973), pp. 463-490.
           "Implementation and Analysis of the Todd-Coxeter Algorithm"

    """
    coset_table_max_limit = 4096000
    coset_table_limit = None
    max_stack_size = 100

    def __init__(self, fp_grp, subgroup, max_cosets=None):
        if False:
            print('Hello World!')
        if not max_cosets:
            max_cosets = CosetTable.coset_table_max_limit
        self.fp_group = fp_grp
        self.subgroup = subgroup
        self.coset_table_limit = max_cosets
        self.p = [0]
        self.A = list(chain.from_iterable(((gen, gen ** (-1)) for gen in self.fp_group.generators)))
        self.P = [[None] * len(self.A)]
        self.table = [[None] * len(self.A)]
        self.A_dict = {x: self.A.index(x) for x in self.A}
        self.A_dict_inv = {}
        for (x, index) in self.A_dict.items():
            if index % 2 == 0:
                self.A_dict_inv[x] = self.A_dict[x] + 1
            else:
                self.A_dict_inv[x] = self.A_dict[x] - 1
        self.deduction_stack = []
        H = self.subgroup
        self._grp = free_group(', '.join(['a_%d' % i for i in range(len(H))]))[0]
        self.P = [[None] * len(self.A)]
        self.p_p = {}

    @property
    def omega(self):
        if False:
            i = 10
            return i + 15
        'Set of live cosets. '
        return [coset for coset in range(len(self.p)) if self.p[coset] == coset]

    def copy(self):
        if False:
            while True:
                i = 10
        '\n        Return a shallow copy of Coset Table instance ``self``.\n\n        '
        self_copy = self.__class__(self.fp_group, self.subgroup)
        self_copy.table = [list(perm_rep) for perm_rep in self.table]
        self_copy.p = list(self.p)
        self_copy.deduction_stack = list(self.deduction_stack)
        return self_copy

    def __str__(self):
        if False:
            return 10
        return 'Coset Table on %s with %s as subgroup generators' % (self.fp_group, self.subgroup)
    __repr__ = __str__

    @property
    def n(self):
        if False:
            print('Hello World!')
        'The number `n` represents the length of the sublist containing the\n        live cosets.\n\n        '
        if not self.table:
            return 0
        return max(self.omega) + 1

    def is_complete(self):
        if False:
            print('Hello World!')
        '\n        The coset table is called complete if it has no undefined entries\n        on the live cosets; that is, `\\alpha^x` is defined for all\n        `\\alpha \\in \\Omega` and `x \\in A`.\n\n        '
        return not any((None in self.table[coset] for coset in self.omega))

    def define(self, alpha, x, modified=False):
        if False:
            while True:
                i = 10
        '\n        This routine is used in the relator-based strategy of Todd-Coxeter\n        algorithm if some `\\alpha^x` is undefined. We check whether there is\n        space available for defining a new coset. If there is enough space\n        then we remedy this by adjoining a new coset `\\beta` to `\\Omega`\n        (i.e to set of live cosets) and put that equal to `\\alpha^x`, then\n        make an assignment satisfying Property[1]. If there is not enough space\n        then we halt the Coset Table creation. The maximum amount of space that\n        can be used by Coset Table can be manipulated using the class variable\n        ``CosetTable.coset_table_max_limit``.\n\n        See Also\n        ========\n\n        define_c\n\n        '
        A = self.A
        table = self.table
        len_table = len(table)
        if len_table >= self.coset_table_limit:
            raise ValueError('the coset enumeration has defined more than %s cosets. Try with a greater value max number of cosets ' % self.coset_table_limit)
        table.append([None] * len(A))
        self.P.append([None] * len(self.A))
        beta = len_table
        self.p.append(beta)
        table[alpha][self.A_dict[x]] = beta
        table[beta][self.A_dict_inv[x]] = alpha
        if modified:
            self.P[alpha][self.A_dict[x]] = self._grp.identity
            self.P[beta][self.A_dict_inv[x]] = self._grp.identity
            self.p_p[beta] = self._grp.identity

    def define_c(self, alpha, x):
        if False:
            while True:
                i = 10
        '\n        A variation of ``define`` routine, described on Pg. 165 [1], used in\n        the coset table-based strategy of Todd-Coxeter algorithm. It differs\n        from ``define`` routine in that for each definition it also adds the\n        tuple `(\\alpha, x)` to the deduction stack.\n\n        See Also\n        ========\n\n        define\n\n        '
        A = self.A
        table = self.table
        len_table = len(table)
        if len_table >= self.coset_table_limit:
            raise ValueError('the coset enumeration has defined more than %s cosets. Try with a greater value max number of cosets ' % self.coset_table_limit)
        table.append([None] * len(A))
        beta = len_table
        self.p.append(beta)
        table[alpha][self.A_dict[x]] = beta
        table[beta][self.A_dict_inv[x]] = alpha
        self.deduction_stack.append((alpha, x))

    def scan_c(self, alpha, word):
        if False:
            for i in range(10):
                print('nop')
        '\n        A variation of ``scan`` routine, described on pg. 165 of [1], which\n        puts at tuple, whenever a deduction occurs, to deduction stack.\n\n        See Also\n        ========\n\n        scan, scan_check, scan_and_fill, scan_and_fill_c\n\n        '
        A_dict = self.A_dict
        A_dict_inv = self.A_dict_inv
        table = self.table
        f = alpha
        i = 0
        r = len(word)
        b = alpha
        j = r - 1
        while i <= j and table[f][A_dict[word[i]]] is not None:
            f = table[f][A_dict[word[i]]]
            i += 1
        if i > j:
            if f != b:
                self.coincidence_c(f, b)
            return
        while j >= i and table[b][A_dict_inv[word[j]]] is not None:
            b = table[b][A_dict_inv[word[j]]]
            j -= 1
        if j < i:
            self.coincidence_c(f, b)
        elif j == i:
            table[f][A_dict[word[i]]] = b
            table[b][A_dict_inv[word[i]]] = f
            self.deduction_stack.append((f, word[i]))

    def coincidence_c(self, alpha, beta):
        if False:
            print('Hello World!')
        '\n        A variation of ``coincidence`` routine used in the coset-table based\n        method of coset enumeration. The only difference being on addition of\n        a new coset in coset table(i.e new coset introduction), then it is\n        appended to ``deduction_stack``.\n\n        See Also\n        ========\n\n        coincidence\n\n        '
        A_dict = self.A_dict
        A_dict_inv = self.A_dict_inv
        table = self.table
        q = []
        self.merge(alpha, beta, q)
        while len(q) > 0:
            gamma = q.pop(0)
            for x in A_dict:
                delta = table[gamma][A_dict[x]]
                if delta is not None:
                    table[delta][A_dict_inv[x]] = None
                    self.deduction_stack.append((delta, x ** (-1)))
                    mu = self.rep(gamma)
                    nu = self.rep(delta)
                    if table[mu][A_dict[x]] is not None:
                        self.merge(nu, table[mu][A_dict[x]], q)
                    elif table[nu][A_dict_inv[x]] is not None:
                        self.merge(mu, table[nu][A_dict_inv[x]], q)
                    else:
                        table[mu][A_dict[x]] = nu
                        table[nu][A_dict_inv[x]] = mu

    def scan(self, alpha, word, y=None, fill=False, modified=False):
        if False:
            print('Hello World!')
        '\n        ``scan`` performs a scanning process on the input ``word``.\n        It first locates the largest prefix ``s`` of ``word`` for which\n        `\\alpha^s` is defined (i.e is not ``None``), ``s`` may be empty. Let\n        ``word=sv``, let ``t`` be the longest suffix of ``v`` for which\n        `\\alpha^{t^{-1}}` is defined, and let ``v=ut``. Then three\n        possibilities are there:\n\n        1. If ``t=v``, then we say that the scan completes, and if, in addition\n        `\\alpha^s = \\alpha^{t^{-1}}`, then we say that the scan completes\n        correctly.\n\n        2. It can also happen that scan does not complete, but `|u|=1`; that\n        is, the word ``u`` consists of a single generator `x \\in A`. In that\n        case, if `\\alpha^s = \\beta` and `\\alpha^{t^{-1}} = \\gamma`, then we can\n        set `\\beta^x = \\gamma` and `\\gamma^{x^{-1}} = \\beta`. These assignments\n        are known as deductions and enable the scan to complete correctly.\n\n        3. See ``coicidence`` routine for explanation of third condition.\n\n        Notes\n        =====\n\n        The code for the procedure of scanning `\\alpha \\in \\Omega`\n        under `w \\in A*` is defined on pg. 155 [1]\n\n        See Also\n        ========\n\n        scan_c, scan_check, scan_and_fill, scan_and_fill_c\n\n        Scan and Fill\n        =============\n\n        Performed when the default argument fill=True.\n\n        Modified Scan\n        =============\n\n        Performed when the default argument modified=True\n\n        '
        A_dict = self.A_dict
        A_dict_inv = self.A_dict_inv
        table = self.table
        f = alpha
        i = 0
        r = len(word)
        b = alpha
        j = r - 1
        b_p = y
        if modified:
            f_p = self._grp.identity
        flag = 0
        while fill or flag == 0:
            flag = 1
            while i <= j and table[f][A_dict[word[i]]] is not None:
                if modified:
                    f_p = f_p * self.P[f][A_dict[word[i]]]
                f = table[f][A_dict[word[i]]]
                i += 1
            if i > j:
                if f != b:
                    if modified:
                        self.modified_coincidence(f, b, f_p ** (-1) * y)
                    else:
                        self.coincidence(f, b)
                return
            while j >= i and table[b][A_dict_inv[word[j]]] is not None:
                if modified:
                    b_p = b_p * self.P[b][self.A_dict_inv[word[j]]]
                b = table[b][A_dict_inv[word[j]]]
                j -= 1
            if j < i:
                if modified:
                    self.modified_coincidence(f, b, f_p ** (-1) * b_p)
                else:
                    self.coincidence(f, b)
            elif j == i:
                table[f][A_dict[word[i]]] = b
                table[b][A_dict_inv[word[i]]] = f
                if modified:
                    self.P[f][self.A_dict[word[i]]] = f_p ** (-1) * b_p
                    self.P[b][self.A_dict_inv[word[i]]] = b_p ** (-1) * f_p
                return
            elif fill:
                self.define(f, word[i], modified=modified)

    def scan_check(self, alpha, word):
        if False:
            return 10
        '\n        Another version of ``scan`` routine, described on, it checks whether\n        `\\alpha` scans correctly under `word`, it is a straightforward\n        modification of ``scan``. ``scan_check`` returns ``False`` (rather than\n        calling ``coincidence``) if the scan completes incorrectly; otherwise\n        it returns ``True``.\n\n        See Also\n        ========\n\n        scan, scan_c, scan_and_fill, scan_and_fill_c\n\n        '
        A_dict = self.A_dict
        A_dict_inv = self.A_dict_inv
        table = self.table
        f = alpha
        i = 0
        r = len(word)
        b = alpha
        j = r - 1
        while i <= j and table[f][A_dict[word[i]]] is not None:
            f = table[f][A_dict[word[i]]]
            i += 1
        if i > j:
            return f == b
        while j >= i and table[b][A_dict_inv[word[j]]] is not None:
            b = table[b][A_dict_inv[word[j]]]
            j -= 1
        if j < i:
            return False
        elif j == i:
            table[f][A_dict[word[i]]] = b
            table[b][A_dict_inv[word[i]]] = f
        return True

    def merge(self, k, lamda, q, w=None, modified=False):
        if False:
            return 10
        "\n        Merge two classes with representatives ``k`` and ``lamda``, described\n        on Pg. 157 [1] (for pseudocode), start by putting ``p[k] = lamda``.\n        It is more efficient to choose the new representative from the larger\n        of the two classes being merged, i.e larger among ``k`` and ``lamda``.\n        procedure ``merge`` performs the merging operation, adds the deleted\n        class representative to the queue ``q``.\n\n        Parameters\n        ==========\n\n        'k', 'lamda' being the two class representatives to be merged.\n\n        Notes\n        =====\n\n        Pg. 86-87 [1] contains a description of this method.\n\n        See Also\n        ========\n\n        coincidence, rep\n\n        "
        p = self.p
        rep = self.rep
        phi = rep(k, modified=modified)
        psi = rep(lamda, modified=modified)
        if phi != psi:
            mu = min(phi, psi)
            v = max(phi, psi)
            p[v] = mu
            if modified:
                if v == phi:
                    self.p_p[phi] = self.p_p[k] ** (-1) * w * self.p_p[lamda]
                else:
                    self.p_p[psi] = self.p_p[lamda] ** (-1) * w ** (-1) * self.p_p[k]
            q.append(v)

    def rep(self, k, modified=False):
        if False:
            return 10
        "\n        Parameters\n        ==========\n\n        `k \\in [0 \\ldots n-1]`, as for ``self`` only array ``p`` is used\n\n        Returns\n        =======\n\n        Representative of the class containing ``k``.\n\n        Returns the representative of `\\sim` class containing ``k``, it also\n        makes some modification to array ``p`` of ``self`` to ease further\n        computations, described on Pg. 157 [1].\n\n        The information on classes under `\\sim` is stored in array `p` of\n        ``self`` argument, which will always satisfy the property:\n\n        `p[\\alpha] \\sim \\alpha` and `p[\\alpha]=\\alpha \\iff \\alpha=rep(\\alpha)`\n        `\\forall \\in [0 \\ldots n-1]`.\n\n        So, for `\\alpha \\in [0 \\ldots n-1]`, we find `rep(self, \\alpha)` by\n        continually replacing `\\alpha` by `p[\\alpha]` until it becomes\n        constant (i.e satisfies `p[\\alpha] = \\alpha`):w\n\n        To increase the efficiency of later ``rep`` calculations, whenever we\n        find `rep(self, \\alpha)=\\beta`, we set\n        `p[\\gamma] = \\beta \\forall \\gamma \\in p-chain` from `\\alpha` to `\\beta`\n\n        Notes\n        =====\n\n        ``rep`` routine is also described on Pg. 85-87 [1] in Atkinson's\n        algorithm, this results from the fact that ``coincidence`` routine\n        introduces functionality similar to that introduced by the\n        ``minimal_block`` routine on Pg. 85-87 [1].\n\n        See Also\n        ========\n\n        coincidence, merge\n\n        "
        p = self.p
        lamda = k
        rho = p[lamda]
        if modified:
            s = p[:]
        while rho != lamda:
            if modified:
                s[rho] = lamda
            lamda = rho
            rho = p[lamda]
        if modified:
            rho = s[lamda]
            while rho != k:
                mu = rho
                rho = s[mu]
                p[rho] = lamda
                self.p_p[rho] = self.p_p[rho] * self.p_p[mu]
        else:
            mu = k
            rho = p[mu]
            while rho != lamda:
                p[mu] = lamda
                mu = rho
                rho = p[mu]
        return lamda

    def coincidence(self, alpha, beta, w=None, modified=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        The third situation described in ``scan`` routine is handled by this\n        routine, described on Pg. 156-161 [1].\n\n        The unfortunate situation when the scan completes but not correctly,\n        then ``coincidence`` routine is run. i.e when for some `i` with\n        `1 \\le i \\le r+1`, we have `w=st` with `s = x_1 x_2 \\dots x_{i-1}`,\n        `t = x_i x_{i+1} \\dots x_r`, and `\\beta = \\alpha^s` and\n        `\\gamma = \\alpha^{t-1}` are defined but unequal. This means that\n        `\\beta` and `\\gamma` represent the same coset of `H` in `G`. Described\n        on Pg. 156 [1]. ``rep``\n\n        See Also\n        ========\n\n        scan\n\n        '
        A_dict = self.A_dict
        A_dict_inv = self.A_dict_inv
        table = self.table
        q = []
        if modified:
            self.modified_merge(alpha, beta, w, q)
        else:
            self.merge(alpha, beta, q)
        while len(q) > 0:
            gamma = q.pop(0)
            for x in A_dict:
                delta = table[gamma][A_dict[x]]
                if delta is not None:
                    table[delta][A_dict_inv[x]] = None
                    mu = self.rep(gamma, modified=modified)
                    nu = self.rep(delta, modified=modified)
                    if table[mu][A_dict[x]] is not None:
                        if modified:
                            v = self.p_p[delta] ** (-1) * self.P[gamma][self.A_dict[x]] ** (-1)
                            v = v * self.p_p[gamma] * self.P[mu][self.A_dict[x]]
                            self.modified_merge(nu, table[mu][self.A_dict[x]], v, q)
                        else:
                            self.merge(nu, table[mu][A_dict[x]], q)
                    elif table[nu][A_dict_inv[x]] is not None:
                        if modified:
                            v = self.p_p[gamma] ** (-1) * self.P[gamma][self.A_dict[x]]
                            v = v * self.p_p[delta] * self.P[mu][self.A_dict_inv[x]]
                            self.modified_merge(mu, table[nu][self.A_dict_inv[x]], v, q)
                        else:
                            self.merge(mu, table[nu][A_dict_inv[x]], q)
                    else:
                        table[mu][A_dict[x]] = nu
                        table[nu][A_dict_inv[x]] = mu
                        if modified:
                            v = self.p_p[gamma] ** (-1) * self.P[gamma][self.A_dict[x]] * self.p_p[delta]
                            self.P[mu][self.A_dict[x]] = v
                            self.P[nu][self.A_dict_inv[x]] = v ** (-1)

    def scan_and_fill(self, alpha, word):
        if False:
            return 10
        '\n        A modified version of ``scan`` routine used in the relator-based\n        method of coset enumeration, described on pg. 162-163 [1], which\n        follows the idea that whenever the procedure is called and the scan\n        is incomplete then it makes new definitions to enable the scan to\n        complete; i.e it fills in the gaps in the scan of the relator or\n        subgroup generator.\n\n        '
        self.scan(alpha, word, fill=True)

    def scan_and_fill_c(self, alpha, word):
        if False:
            i = 10
            return i + 15
        '\n        A modified version of ``scan`` routine, described on Pg. 165 second\n        para. [1], with modification similar to that of ``scan_anf_fill`` the\n        only difference being it calls the coincidence procedure used in the\n        coset-table based method i.e. the routine ``coincidence_c`` is used.\n\n        See Also\n        ========\n\n        scan, scan_and_fill\n\n        '
        A_dict = self.A_dict
        A_dict_inv = self.A_dict_inv
        table = self.table
        r = len(word)
        f = alpha
        i = 0
        b = alpha
        j = r - 1
        while True:
            while i <= j and table[f][A_dict[word[i]]] is not None:
                f = table[f][A_dict[word[i]]]
                i += 1
            if i > j:
                if f != b:
                    self.coincidence_c(f, b)
                return
            while j >= i and table[b][A_dict_inv[word[j]]] is not None:
                b = table[b][A_dict_inv[word[j]]]
                j -= 1
            if j < i:
                self.coincidence_c(f, b)
            elif j == i:
                table[f][A_dict[word[i]]] = b
                table[b][A_dict_inv[word[i]]] = f
                self.deduction_stack.append((f, word[i]))
            else:
                self.define_c(f, word[i])

    def look_ahead(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        When combined with the HLT method this is known as HLT+Lookahead\n        method of coset enumeration, described on pg. 164 [1]. Whenever\n        ``define`` aborts due to lack of space available this procedure is\n        executed. This routine helps in recovering space resulting from\n        "coincidence" of cosets.\n\n        '
        R = self.fp_group.relators
        p = self.p
        for beta in self.omega:
            for w in R:
                self.scan(beta, w)
                if p[beta] < beta:
                    break

    def process_deductions(self, R_c_x, R_c_x_inv):
        if False:
            return 10
        '\n        Processes the deductions that have been pushed onto ``deduction_stack``,\n        described on Pg. 166 [1] and is used in coset-table based enumeration.\n\n        See Also\n        ========\n\n        deduction_stack\n\n        '
        p = self.p
        table = self.table
        while len(self.deduction_stack) > 0:
            if len(self.deduction_stack) >= CosetTable.max_stack_size:
                self.look_ahead()
                del self.deduction_stack[:]
                continue
            else:
                (alpha, x) = self.deduction_stack.pop()
                if p[alpha] == alpha:
                    for w in R_c_x:
                        self.scan_c(alpha, w)
                        if p[alpha] < alpha:
                            break
            beta = table[alpha][self.A_dict[x]]
            if beta is not None and p[beta] == beta:
                for w in R_c_x_inv:
                    self.scan_c(beta, w)
                    if p[beta] < beta:
                        break

    def process_deductions_check(self, R_c_x, R_c_x_inv):
        if False:
            return 10
        '\n        A variation of ``process_deductions``, this calls ``scan_check``\n        wherever ``process_deductions`` calls ``scan``, described on Pg. [1].\n\n        See Also\n        ========\n\n        process_deductions\n\n        '
        table = self.table
        while len(self.deduction_stack) > 0:
            (alpha, x) = self.deduction_stack.pop()
            for w in R_c_x:
                if not self.scan_check(alpha, w):
                    return False
            beta = table[alpha][self.A_dict[x]]
            if beta is not None:
                for w in R_c_x_inv:
                    if not self.scan_check(beta, w):
                        return False
        return True

    def switch(self, beta, gamma):
        if False:
            return 10
        'Switch the elements `\\beta, \\gamma \\in \\Omega` of ``self``, used\n        by the ``standardize`` procedure, described on Pg. 167 [1].\n\n        See Also\n        ========\n\n        standardize\n\n        '
        A = self.A
        A_dict = self.A_dict
        table = self.table
        for x in A:
            z = table[gamma][A_dict[x]]
            table[gamma][A_dict[x]] = table[beta][A_dict[x]]
            table[beta][A_dict[x]] = z
            for alpha in range(len(self.p)):
                if self.p[alpha] == alpha:
                    if table[alpha][A_dict[x]] == beta:
                        table[alpha][A_dict[x]] = gamma
                    elif table[alpha][A_dict[x]] == gamma:
                        table[alpha][A_dict[x]] = beta

    def standardize(self):
        if False:
            print('Hello World!')
        '\n        A coset table is standardized if when running through the cosets and\n        within each coset through the generator images (ignoring generator\n        inverses), the cosets appear in order of the integers\n        `0, 1, \\dots, n`. "Standardize" reorders the elements of `\\Omega`\n        such that, if we scan the coset table first by elements of `\\Omega`\n        and then by elements of A, then the cosets occur in ascending order.\n        ``standardize()`` is used at the end of an enumeration to permute the\n        cosets so that they occur in some sort of standard order.\n\n        Notes\n        =====\n\n        procedure is described on pg. 167-168 [1], it also makes use of the\n        ``switch`` routine to replace by smaller integer value.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> from sympy.combinatorics.fp_groups import FpGroup, coset_enumeration_r\n        >>> F, x, y = free_group("x, y")\n\n        # Example 5.3 from [1]\n        >>> f = FpGroup(F, [x**2*y**2, x**3*y**5])\n        >>> C = coset_enumeration_r(f, [])\n        >>> C.compress()\n        >>> C.table\n        [[1, 3, 1, 3], [2, 0, 2, 0], [3, 1, 3, 1], [0, 2, 0, 2]]\n        >>> C.standardize()\n        >>> C.table\n        [[1, 2, 1, 2], [3, 0, 3, 0], [0, 3, 0, 3], [2, 1, 2, 1]]\n\n        '
        A = self.A
        A_dict = self.A_dict
        gamma = 1
        for (alpha, x) in product(range(self.n), A):
            beta = self.table[alpha][A_dict[x]]
            if beta >= gamma:
                if beta > gamma:
                    self.switch(gamma, beta)
                gamma += 1
                if gamma == self.n:
                    return

    def compress(self):
        if False:
            i = 10
            return i + 15
        'Removes the non-live cosets from the coset table, described on\n        pg. 167 [1].\n\n        '
        gamma = -1
        A = self.A
        A_dict = self.A_dict
        A_dict_inv = self.A_dict_inv
        table = self.table
        chi = tuple([i for i in range(len(self.p)) if self.p[i] != i])
        for alpha in self.omega:
            gamma += 1
            if gamma != alpha:
                for x in A:
                    beta = table[alpha][A_dict[x]]
                    table[gamma][A_dict[x]] = beta
                    table[beta][A_dict_inv[x]] == gamma
        self.p = list(range(gamma + 1))
        del table[len(self.p):]
        for row in table:
            for j in range(len(self.A)):
                row[j] -= bisect_left(chi, row[j])

    def conjugates(self, R):
        if False:
            for i in range(10):
                print('nop')
        R_c = list(chain.from_iterable(((rel.cyclic_conjugates(), (rel ** (-1)).cyclic_conjugates()) for rel in R)))
        R_set = set()
        for conjugate in R_c:
            R_set = R_set.union(conjugate)
        R_c_list = []
        for x in self.A:
            r = {word for word in R_set if word[0] == x}
            R_c_list.append(r)
            R_set.difference_update(r)
        return R_c_list

    def coset_representative(self, coset):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute the coset representative of a given coset.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> from sympy.combinatorics.fp_groups import FpGroup, coset_enumeration_r\n        >>> F, x, y = free_group("x, y")\n        >>> f = FpGroup(F, [x**3, y**3, x**-1*y**-1*x*y])\n        >>> C = coset_enumeration_r(f, [x])\n        >>> C.compress()\n        >>> C.table\n        [[0, 0, 1, 2], [1, 1, 2, 0], [2, 2, 0, 1]]\n        >>> C.coset_representative(0)\n        <identity>\n        >>> C.coset_representative(1)\n        y\n        >>> C.coset_representative(2)\n        y**-1\n\n        '
        for x in self.A:
            gamma = self.table[coset][self.A_dict[x]]
            if coset == 0:
                return self.fp_group.identity
            if gamma < coset:
                return self.coset_representative(gamma) * x ** (-1)

    def modified_define(self, alpha, x):
        if False:
            for i in range(10):
                print('nop')
        '\n        Define a function p_p from from [1..n] to A* as\n        an additional component of the modified coset table.\n\n        Parameters\n        ==========\n\n        \\alpha \\in \\Omega\n        x \\in A*\n\n        See Also\n        ========\n\n        define\n\n        '
        self.define(alpha, x, modified=True)

    def modified_scan(self, alpha, w, y, fill=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ==========\n        \\alpha \\in \\Omega\n        w \\in A*\n        y \\in (YUY^-1)\n        fill -- `modified_scan_and_fill` when set to True.\n\n        See Also\n        ========\n\n        scan\n        '
        self.scan(alpha, w, y=y, fill=fill, modified=True)

    def modified_scan_and_fill(self, alpha, w, y):
        if False:
            return 10
        self.modified_scan(alpha, w, y, fill=True)

    def modified_merge(self, k, lamda, w, q):
        if False:
            i = 10
            return i + 15
        "\n        Parameters\n        ==========\n\n        'k', 'lamda' -- the two class representatives to be merged.\n        q -- queue of length l of elements to be deleted from `\\Omega` *.\n        w -- Word in (YUY^-1)\n\n        See Also\n        ========\n\n        merge\n        "
        self.merge(k, lamda, q, w=w, modified=True)

    def modified_rep(self, k):
        if False:
            print('Hello World!')
        '\n        Parameters\n        ==========\n\n        `k \\in [0 \\ldots n-1]`\n\n        See Also\n        ========\n\n        rep\n        '
        self.rep(k, modified=True)

    def modified_coincidence(self, alpha, beta, w):
        if False:
            i = 10
            return i + 15
        '\n        Parameters\n        ==========\n\n        A coincident pair `\\alpha, \\beta \\in \\Omega, w \\in Y \\cup Y^{-1}`\n\n        See Also\n        ========\n\n        coincidence\n\n        '
        self.coincidence(alpha, beta, w=w, modified=True)

def coset_enumeration_r(fp_grp, Y, max_cosets=None, draft=None, incomplete=False, modified=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    This is easier of the two implemented methods of coset enumeration.\n    and is often called the HLT method, after Hazelgrove, Leech, Trotter\n    The idea is that we make use of ``scan_and_fill`` makes new definitions\n    whenever the scan is incomplete to enable the scan to complete; this way\n    we fill in the gaps in the scan of the relator or subgroup generator,\n    that\'s why the name relator-based method.\n\n    An instance of `CosetTable` for `fp_grp` can be passed as the keyword\n    argument `draft` in which case the coset enumeration will start with\n    that instance and attempt to complete it.\n\n    When `incomplete` is `True` and the function is unable to complete for\n    some reason, the partially complete table will be returned.\n\n    # TODO: complete the docstring\n\n    See Also\n    ========\n\n    scan_and_fill,\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.free_groups import free_group\n    >>> from sympy.combinatorics.fp_groups import FpGroup, coset_enumeration_r\n    >>> F, x, y = free_group("x, y")\n\n    # Example 5.1 from [1]\n    >>> f = FpGroup(F, [x**3, y**3, x**-1*y**-1*x*y])\n    >>> C = coset_enumeration_r(f, [x])\n    >>> for i in range(len(C.p)):\n    ...     if C.p[i] == i:\n    ...         print(C.table[i])\n    [0, 0, 1, 2]\n    [1, 1, 2, 0]\n    [2, 2, 0, 1]\n    >>> C.p\n    [0, 1, 2, 1, 1]\n\n    # Example from exercises Q2 [1]\n    >>> f = FpGroup(F, [x**2*y**2, y**-1*x*y*x**-3])\n    >>> C = coset_enumeration_r(f, [])\n    >>> C.compress(); C.standardize()\n    >>> C.table\n    [[1, 2, 3, 4],\n    [5, 0, 6, 7],\n    [0, 5, 7, 6],\n    [7, 6, 5, 0],\n    [6, 7, 0, 5],\n    [2, 1, 4, 3],\n    [3, 4, 2, 1],\n    [4, 3, 1, 2]]\n\n    # Example 5.2\n    >>> f = FpGroup(F, [x**2, y**3, (x*y)**3])\n    >>> Y = [x*y]\n    >>> C = coset_enumeration_r(f, Y)\n    >>> for i in range(len(C.p)):\n    ...     if C.p[i] == i:\n    ...         print(C.table[i])\n    [1, 1, 2, 1]\n    [0, 0, 0, 2]\n    [3, 3, 1, 0]\n    [2, 2, 3, 3]\n\n    # Example 5.3\n    >>> f = FpGroup(F, [x**2*y**2, x**3*y**5])\n    >>> Y = []\n    >>> C = coset_enumeration_r(f, Y)\n    >>> for i in range(len(C.p)):\n    ...     if C.p[i] == i:\n    ...         print(C.table[i])\n    [1, 3, 1, 3]\n    [2, 0, 2, 0]\n    [3, 1, 3, 1]\n    [0, 2, 0, 2]\n\n    # Example 5.4\n    >>> F, a, b, c, d, e = free_group("a, b, c, d, e")\n    >>> f = FpGroup(F, [a*b*c**-1, b*c*d**-1, c*d*e**-1, d*e*a**-1, e*a*b**-1])\n    >>> Y = [a]\n    >>> C = coset_enumeration_r(f, Y)\n    >>> for i in range(len(C.p)):\n    ...     if C.p[i] == i:\n    ...         print(C.table[i])\n    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]\n\n    # example of "compress" method\n    >>> C.compress()\n    >>> C.table\n    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]\n\n    # Exercises Pg. 161, Q2.\n    >>> F, x, y = free_group("x, y")\n    >>> f = FpGroup(F, [x**2*y**2, y**-1*x*y*x**-3])\n    >>> Y = []\n    >>> C = coset_enumeration_r(f, Y)\n    >>> C.compress()\n    >>> C.standardize()\n    >>> C.table\n    [[1, 2, 3, 4],\n    [5, 0, 6, 7],\n    [0, 5, 7, 6],\n    [7, 6, 5, 0],\n    [6, 7, 0, 5],\n    [2, 1, 4, 3],\n    [3, 4, 2, 1],\n    [4, 3, 1, 2]]\n\n    # John J. Cannon; Lucien A. Dimino; George Havas; Jane M. Watson\n    # Mathematics of Computation, Vol. 27, No. 123. (Jul., 1973), pp. 463-490\n    # from 1973chwd.pdf\n    # Table 1. Ex. 1\n    >>> F, r, s, t = free_group("r, s, t")\n    >>> E1 = FpGroup(F, [t**-1*r*t*r**-2, r**-1*s*r*s**-2, s**-1*t*s*t**-2])\n    >>> C = coset_enumeration_r(E1, [r])\n    >>> for i in range(len(C.p)):\n    ...     if C.p[i] == i:\n    ...         print(C.table[i])\n    [0, 0, 0, 0, 0, 0]\n\n    Ex. 2\n    >>> F, a, b = free_group("a, b")\n    >>> Cox = FpGroup(F, [a**6, b**6, (a*b)**2, (a**2*b**2)**2, (a**3*b**3)**5])\n    >>> C = coset_enumeration_r(Cox, [a])\n    >>> index = 0\n    >>> for i in range(len(C.p)):\n    ...     if C.p[i] == i:\n    ...         index += 1\n    >>> index\n    500\n\n    # Ex. 3\n    >>> F, a, b = free_group("a, b")\n    >>> B_2_4 = FpGroup(F, [a**4, b**4, (a*b)**4, (a**-1*b)**4, (a**2*b)**4,             (a*b**2)**4, (a**2*b**2)**4, (a**-1*b*a*b)**4, (a*b**-1*a*b)**4])\n    >>> C = coset_enumeration_r(B_2_4, [a])\n    >>> index = 0\n    >>> for i in range(len(C.p)):\n    ...     if C.p[i] == i:\n    ...         index += 1\n    >>> index\n    1024\n\n    References\n    ==========\n\n    .. [1] Holt, D., Eick, B., O\'Brien, E.\n           "Handbook of computational group theory"\n\n    '
    C = CosetTable(fp_grp, Y, max_cosets=max_cosets)
    if modified:
        _scan_and_fill = C.modified_scan_and_fill
        _define = C.modified_define
    else:
        _scan_and_fill = C.scan_and_fill
        _define = C.define
    if draft:
        C.table = draft.table[:]
        C.p = draft.p[:]
    R = fp_grp.relators
    A_dict = C.A_dict
    p = C.p
    for i in range(len(Y)):
        if modified:
            _scan_and_fill(0, Y[i], C._grp.generators[i])
        else:
            _scan_and_fill(0, Y[i])
    alpha = 0
    while alpha < C.n:
        if p[alpha] == alpha:
            try:
                for w in R:
                    if modified:
                        _scan_and_fill(alpha, w, C._grp.identity)
                    else:
                        _scan_and_fill(alpha, w)
                    if p[alpha] < alpha:
                        break
                if p[alpha] == alpha:
                    for x in A_dict:
                        if C.table[alpha][A_dict[x]] is None:
                            _define(alpha, x)
            except ValueError as e:
                if incomplete:
                    return C
                raise e
        alpha += 1
    return C

def modified_coset_enumeration_r(fp_grp, Y, max_cosets=None, draft=None, incomplete=False):
    if False:
        i = 10
        return i + 15
    '\n    Introduce a new set of symbols y \\in Y that correspond to the\n    generators of the subgroup. Store the elements of Y as a\n    word P[\\alpha, x] and compute the coset table similar to that of\n    the regular coset enumeration methods.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.free_groups import free_group\n    >>> from sympy.combinatorics.fp_groups import FpGroup\n    >>> from sympy.combinatorics.coset_table import modified_coset_enumeration_r\n    >>> F, x, y = free_group("x, y")\n    >>> f = FpGroup(F, [x**3, y**3, x**-1*y**-1*x*y])\n    >>> C = modified_coset_enumeration_r(f, [x])\n    >>> C.table\n    [[0, 0, 1, 2], [1, 1, 2, 0], [2, 2, 0, 1], [None, 1, None, None], [1, 3, None, None]]\n\n    See Also\n    ========\n\n    coset_enumertation_r\n\n    References\n    ==========\n\n    .. [1] Holt, D., Eick, B., O\'Brien, E.,\n           "Handbook of Computational Group Theory",\n           Section 5.3.2\n    '
    return coset_enumeration_r(fp_grp, Y, max_cosets=max_cosets, draft=draft, incomplete=incomplete, modified=True)

def coset_enumeration_c(fp_grp, Y, max_cosets=None, draft=None, incomplete=False):
    if False:
        while True:
            i = 10
    '\n    >>> from sympy.combinatorics.free_groups import free_group\n    >>> from sympy.combinatorics.fp_groups import FpGroup, coset_enumeration_c\n    >>> F, x, y = free_group("x, y")\n    >>> f = FpGroup(F, [x**3, y**3, x**-1*y**-1*x*y])\n    >>> C = coset_enumeration_c(f, [x])\n    >>> C.table\n    [[0, 0, 1, 2], [1, 1, 2, 0], [2, 2, 0, 1]]\n\n    '
    X = fp_grp.generators
    R = fp_grp.relators
    C = CosetTable(fp_grp, Y, max_cosets=max_cosets)
    if draft:
        C.table = draft.table[:]
        C.p = draft.p[:]
        C.deduction_stack = draft.deduction_stack
        for (alpha, x) in product(range(len(C.table)), X):
            if C.table[alpha][C.A_dict[x]] is not None:
                C.deduction_stack.append((alpha, x))
    A = C.A
    R_cyc_red = [rel.identity_cyclic_reduction() for rel in R]
    R_c = list(chain.from_iterable(((rel.cyclic_conjugates(), (rel ** (-1)).cyclic_conjugates()) for rel in R_cyc_red)))
    R_set = set()
    for conjugate in R_c:
        R_set = R_set.union(conjugate)
    R_c_list = []
    for x in C.A:
        r = {word for word in R_set if word[0] == x}
        R_c_list.append(r)
        R_set.difference_update(r)
    for w in Y:
        C.scan_and_fill_c(0, w)
    for x in A:
        C.process_deductions(R_c_list[C.A_dict[x]], R_c_list[C.A_dict_inv[x]])
    alpha = 0
    while alpha < len(C.table):
        if C.p[alpha] == alpha:
            try:
                for x in C.A:
                    if C.p[alpha] != alpha:
                        break
                    if C.table[alpha][C.A_dict[x]] is None:
                        C.define_c(alpha, x)
                        C.process_deductions(R_c_list[C.A_dict[x]], R_c_list[C.A_dict_inv[x]])
            except ValueError as e:
                if incomplete:
                    return C
                raise e
        alpha += 1
    return C