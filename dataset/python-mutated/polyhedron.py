from sympy.combinatorics import Permutation as Perm
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.core import Basic, Tuple, default_sort_key
from sympy.sets import FiniteSet
from sympy.utilities.iterables import minlex, unflatten, flatten
from sympy.utilities.misc import as_int
rmul = Perm.rmul

class Polyhedron(Basic):
    """
    Represents the polyhedral symmetry group (PSG).

    Explanation
    ===========

    The PSG is one of the symmetry groups of the Platonic solids.
    There are three polyhedral groups: the tetrahedral group
    of order 12, the octahedral group of order 24, and the
    icosahedral group of order 60.

    All doctests have been given in the docstring of the
    constructor of the object.

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PolyhedralGroup.html

    """
    _edges = None

    def __new__(cls, corners, faces=(), pgroup=()):
        if False:
            print('Hello World!')
        '\n        The constructor of the Polyhedron group object.\n\n        Explanation\n        ===========\n\n        It takes up to three parameters: the corners, faces, and\n        allowed transformations.\n\n        The corners/vertices are entered as a list of arbitrary\n        expressions that are used to identify each vertex.\n\n        The faces are entered as a list of tuples of indices; a tuple\n        of indices identifies the vertices which define the face. They\n        should be entered in a cw or ccw order; they will be standardized\n        by reversal and rotation to be give the lowest lexical ordering.\n        If no faces are given then no edges will be computed.\n\n            >>> from sympy.combinatorics.polyhedron import Polyhedron\n            >>> Polyhedron(list(\'abc\'), [(1, 2, 0)]).faces\n            {(0, 1, 2)}\n            >>> Polyhedron(list(\'abc\'), [(1, 0, 2)]).faces\n            {(0, 1, 2)}\n\n        The allowed transformations are entered as allowable permutations\n        of the vertices for the polyhedron. Instance of Permutations\n        (as with faces) should refer to the supplied vertices by index.\n        These permutation are stored as a PermutationGroup.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.permutations import Permutation\n        >>> from sympy import init_printing\n        >>> from sympy.abc import w, x, y, z\n        >>> init_printing(pretty_print=False, perm_cyclic=False)\n\n        Here we construct the Polyhedron object for a tetrahedron.\n\n        >>> corners = [w, x, y, z]\n        >>> faces = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 2, 3)]\n\n        Next, allowed transformations of the polyhedron must be given. This\n        is given as permutations of vertices.\n\n        Although the vertices of a tetrahedron can be numbered in 24 (4!)\n        different ways, there are only 12 different orientations for a\n        physical tetrahedron. The following permutations, applied once or\n        twice, will generate all 12 of the orientations. (The identity\n        permutation, Permutation(range(4)), is not included since it does\n        not change the orientation of the vertices.)\n\n        >>> pgroup = [Permutation([[0, 1, 2], [3]]),                       Permutation([[0, 1, 3], [2]]),                       Permutation([[0, 2, 3], [1]]),                       Permutation([[1, 2, 3], [0]]),                       Permutation([[0, 1], [2, 3]]),                       Permutation([[0, 2], [1, 3]]),                       Permutation([[0, 3], [1, 2]])]\n\n        The Polyhedron is now constructed and demonstrated:\n\n        >>> tetra = Polyhedron(corners, faces, pgroup)\n        >>> tetra.size\n        4\n        >>> tetra.edges\n        {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}\n        >>> tetra.corners\n        (w, x, y, z)\n\n        It can be rotated with an arbitrary permutation of vertices, e.g.\n        the following permutation is not in the pgroup:\n\n        >>> tetra.rotate(Permutation([0, 1, 3, 2]))\n        >>> tetra.corners\n        (w, x, z, y)\n\n        An allowed permutation of the vertices can be constructed by\n        repeatedly applying permutations from the pgroup to the vertices.\n        Here is a demonstration that applying p and p**2 for every p in\n        pgroup generates all the orientations of a tetrahedron and no others:\n\n        >>> all = ( (w, x, y, z),                     (x, y, w, z),                     (y, w, x, z),                     (w, z, x, y),                     (z, w, y, x),                     (w, y, z, x),                     (y, z, w, x),                     (x, z, y, w),                     (z, y, x, w),                     (y, x, z, w),                     (x, w, z, y),                     (z, x, w, y) )\n\n        >>> got = []\n        >>> for p in (pgroup + [p**2 for p in pgroup]):\n        ...     h = Polyhedron(corners)\n        ...     h.rotate(p)\n        ...     got.append(h.corners)\n        ...\n        >>> set(got) == set(all)\n        True\n\n        The make_perm method of a PermutationGroup will randomly pick\n        permutations, multiply them together, and return the permutation that\n        can be applied to the polyhedron to give the orientation produced\n        by those individual permutations.\n\n        Here, 3 permutations are used:\n\n        >>> tetra.pgroup.make_perm(3) # doctest: +SKIP\n        Permutation([0, 3, 1, 2])\n\n        To select the permutations that should be used, supply a list\n        of indices to the permutations in pgroup in the order they should\n        be applied:\n\n        >>> use = [0, 0, 2]\n        >>> p002 = tetra.pgroup.make_perm(3, use)\n        >>> p002\n        Permutation([1, 0, 3, 2])\n\n\n        Apply them one at a time:\n\n        >>> tetra.reset()\n        >>> for i in use:\n        ...     tetra.rotate(pgroup[i])\n        ...\n        >>> tetra.vertices\n        (x, w, z, y)\n        >>> sequentially = tetra.vertices\n\n        Apply the composite permutation:\n\n        >>> tetra.reset()\n        >>> tetra.rotate(p002)\n        >>> tetra.corners\n        (x, w, z, y)\n        >>> tetra.corners in all and tetra.corners == sequentially\n        True\n\n        Notes\n        =====\n\n        Defining permutation groups\n        ---------------------------\n\n        It is not necessary to enter any permutations, nor is necessary to\n        enter a complete set of transformations. In fact, for a polyhedron,\n        all configurations can be constructed from just two permutations.\n        For example, the orientations of a tetrahedron can be generated from\n        an axis passing through a vertex and face and another axis passing\n        through a different vertex or from an axis passing through the\n        midpoints of two edges opposite of each other.\n\n        For simplicity of presentation, consider a square --\n        not a cube -- with vertices 1, 2, 3, and 4:\n\n        1-----2  We could think of axes of rotation being:\n        |     |  1) through the face\n        |     |  2) from midpoint 1-2 to 3-4 or 1-3 to 2-4\n        3-----4  3) lines 1-4 or 2-3\n\n\n        To determine how to write the permutations, imagine 4 cameras,\n        one at each corner, labeled A-D:\n\n        A       B          A       B\n         1-----2            1-----3             vertex index:\n         |     |            |     |                 1   0\n         |     |            |     |                 2   1\n         3-----4            2-----4                 3   2\n        C       D          C       D                4   3\n\n        original           after rotation\n                           along 1-4\n\n        A diagonal and a face axis will be chosen for the "permutation group"\n        from which any orientation can be constructed.\n\n        >>> pgroup = []\n\n        Imagine a clockwise rotation when viewing 1-4 from camera A. The new\n        orientation is (in camera-order): 1, 3, 2, 4 so the permutation is\n        given using the *indices* of the vertices as:\n\n        >>> pgroup.append(Permutation((0, 2, 1, 3)))\n\n        Now imagine rotating clockwise when looking down an axis entering the\n        center of the square as viewed. The new camera-order would be\n        3, 1, 4, 2 so the permutation is (using indices):\n\n        >>> pgroup.append(Permutation((2, 0, 3, 1)))\n\n        The square can now be constructed:\n            ** use real-world labels for the vertices, entering them in\n               camera order\n            ** for the faces we use zero-based indices of the vertices\n               in *edge-order* as the face is traversed; neither the\n               direction nor the starting point matter -- the faces are\n               only used to define edges (if so desired).\n\n        >>> square = Polyhedron((1, 2, 3, 4), [(0, 1, 3, 2)], pgroup)\n\n        To rotate the square with a single permutation we can do:\n\n        >>> square.rotate(square.pgroup[0])\n        >>> square.corners\n        (1, 3, 2, 4)\n\n        To use more than one permutation (or to use one permutation more\n        than once) it is more convenient to use the make_perm method:\n\n        >>> p011 = square.pgroup.make_perm([0, 1, 1]) # diag flip + 2 rotations\n        >>> square.reset() # return to initial orientation\n        >>> square.rotate(p011)\n        >>> square.corners\n        (4, 2, 3, 1)\n\n        Thinking outside the box\n        ------------------------\n\n        Although the Polyhedron object has a direct physical meaning, it\n        actually has broader application. In the most general sense it is\n        just a decorated PermutationGroup, allowing one to connect the\n        permutations to something physical. For example, a Rubik\'s cube is\n        not a proper polyhedron, but the Polyhedron class can be used to\n        represent it in a way that helps to visualize the Rubik\'s cube.\n\n        >>> from sympy import flatten, unflatten, symbols\n        >>> from sympy.combinatorics import RubikGroup\n        >>> facelets = flatten([symbols(s+\'1:5\') for s in \'UFRBLD\'])\n        >>> def show():\n        ...     pairs = unflatten(r2.corners, 2)\n        ...     print(pairs[::2])\n        ...     print(pairs[1::2])\n        ...\n        >>> r2 = Polyhedron(facelets, pgroup=RubikGroup(2))\n        >>> show()\n        [(U1, U2), (F1, F2), (R1, R2), (B1, B2), (L1, L2), (D1, D2)]\n        [(U3, U4), (F3, F4), (R3, R4), (B3, B4), (L3, L4), (D3, D4)]\n        >>> r2.rotate(0) # cw rotation of F\n        >>> show()\n        [(U1, U2), (F3, F1), (U3, R2), (B1, B2), (L1, D1), (R3, R1)]\n        [(L4, L2), (F4, F2), (U4, R4), (B3, B4), (L3, D2), (D3, D4)]\n\n        Predefined Polyhedra\n        ====================\n\n        For convenience, the vertices and faces are defined for the following\n        standard solids along with a permutation group for transformations.\n        When the polyhedron is oriented as indicated below, the vertices in\n        a given horizontal plane are numbered in ccw direction, starting from\n        the vertex that will give the lowest indices in a given face. (In the\n        net of the vertices, indices preceded by "-" indicate replication of\n        the lhs index in the net.)\n\n        tetrahedron, tetrahedron_faces\n        ------------------------------\n\n            4 vertices (vertex up) net:\n\n                 0 0-0\n                1 2 3-1\n\n            4 faces:\n\n            (0, 1, 2) (0, 2, 3) (0, 3, 1) (1, 2, 3)\n\n        cube, cube_faces\n        ----------------\n\n            8 vertices (face up) net:\n\n                0 1 2 3-0\n                4 5 6 7-4\n\n            6 faces:\n\n            (0, 1, 2, 3)\n            (0, 1, 5, 4) (1, 2, 6, 5) (2, 3, 7, 6) (0, 3, 7, 4)\n            (4, 5, 6, 7)\n\n        octahedron, octahedron_faces\n        ----------------------------\n\n            6 vertices (vertex up) net:\n\n                 0 0 0-0\n                1 2 3 4-1\n                 5 5 5-5\n\n            8 faces:\n\n            (0, 1, 2) (0, 2, 3) (0, 3, 4) (0, 1, 4)\n            (1, 2, 5) (2, 3, 5) (3, 4, 5) (1, 4, 5)\n\n        dodecahedron, dodecahedron_faces\n        --------------------------------\n\n            20 vertices (vertex up) net:\n\n                  0  1  2  3  4 -0\n                  5  6  7  8  9 -5\n                14 10 11 12 13-14\n                15 16 17 18 19-15\n\n            12 faces:\n\n            (0, 1, 2, 3, 4) (0, 1, 6, 10, 5) (1, 2, 7, 11, 6)\n            (2, 3, 8, 12, 7) (3, 4, 9, 13, 8) (0, 4, 9, 14, 5)\n            (5, 10, 16, 15, 14) (6, 10, 16, 17, 11) (7, 11, 17, 18, 12)\n            (8, 12, 18, 19, 13) (9, 13, 19, 15, 14)(15, 16, 17, 18, 19)\n\n        icosahedron, icosahedron_faces\n        ------------------------------\n\n            12 vertices (face up) net:\n\n                 0  0  0  0 -0\n                1  2  3  4  5 -1\n                 6  7  8  9  10 -6\n                  11 11 11 11 -11\n\n            20 faces:\n\n            (0, 1, 2) (0, 2, 3) (0, 3, 4)\n            (0, 4, 5) (0, 1, 5) (1, 2, 6)\n            (2, 3, 7) (3, 4, 8) (4, 5, 9)\n            (1, 5, 10) (2, 6, 7) (3, 7, 8)\n            (4, 8, 9) (5, 9, 10) (1, 6, 10)\n            (6, 7, 11) (7, 8, 11) (8, 9, 11)\n            (9, 10, 11) (6, 10, 11)\n\n        >>> from sympy.combinatorics.polyhedron import cube\n        >>> cube.edges\n        {(0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7)}\n\n        If you want to use letters or other names for the corners you\n        can still use the pre-calculated faces:\n\n        >>> corners = list(\'abcdefgh\')\n        >>> Polyhedron(corners, cube.faces).corners\n        (a, b, c, d, e, f, g, h)\n\n        References\n        ==========\n\n        .. [1] www.ocf.berkeley.edu/~wwu/articles/platonicsolids.pdf\n\n        '
        faces = [minlex(f, directed=False, key=default_sort_key) for f in faces]
        (corners, faces, pgroup) = args = [Tuple(*a) for a in (corners, faces, pgroup)]
        obj = Basic.__new__(cls, *args)
        obj._corners = tuple(corners)
        obj._faces = FiniteSet(*faces)
        if pgroup and pgroup[0].size != len(corners):
            raise ValueError('Permutation size unequal to number of corners.')
        obj._pgroup = PermutationGroup(pgroup or [Perm(range(len(corners)))])
        return obj

    @property
    def corners(self):
        if False:
            i = 10
            return i + 15
        "\n        Get the corners of the Polyhedron.\n\n        The method ``vertices`` is an alias for ``corners``.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Polyhedron\n        >>> from sympy.abc import a, b, c, d\n        >>> p = Polyhedron(list('abcd'))\n        >>> p.corners == p.vertices == (a, b, c, d)\n        True\n\n        See Also\n        ========\n\n        array_form, cyclic_form\n        "
        return self._corners
    vertices = corners

    @property
    def array_form(self):
        if False:
            print('Hello World!')
        'Return the indices of the corners.\n\n        The indices are given relative to the original position of corners.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.polyhedron import tetrahedron\n        >>> tetrahedron = tetrahedron.copy()\n        >>> tetrahedron.array_form\n        [0, 1, 2, 3]\n\n        >>> tetrahedron.rotate(0)\n        >>> tetrahedron.array_form\n        [0, 2, 3, 1]\n        >>> tetrahedron.pgroup[0].array_form\n        [0, 2, 3, 1]\n\n        See Also\n        ========\n\n        corners, cyclic_form\n        '
        corners = list(self.args[0])
        return [corners.index(c) for c in self.corners]

    @property
    def cyclic_form(self):
        if False:
            i = 10
            return i + 15
        'Return the indices of the corners in cyclic notation.\n\n        The indices are given relative to the original position of corners.\n\n        See Also\n        ========\n\n        corners, array_form\n        '
        return Perm._af_new(self.array_form).cyclic_form

    @property
    def size(self):
        if False:
            return 10
        '\n        Get the number of corners of the Polyhedron.\n        '
        return len(self._corners)

    @property
    def faces(self):
        if False:
            print('Hello World!')
        '\n        Get the faces of the Polyhedron.\n        '
        return self._faces

    @property
    def pgroup(self):
        if False:
            i = 10
            return i + 15
        '\n        Get the permutations of the Polyhedron.\n        '
        return self._pgroup

    @property
    def edges(self):
        if False:
            print('Hello World!')
        '\n        Given the faces of the polyhedra we can get the edges.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Polyhedron\n        >>> from sympy.abc import a, b, c\n        >>> corners = (a, b, c)\n        >>> faces = [(0, 1, 2)]\n        >>> Polyhedron(corners, faces).edges\n        {(0, 1), (0, 2), (1, 2)}\n\n        '
        if self._edges is None:
            output = set()
            for face in self.faces:
                for i in range(len(face)):
                    edge = tuple(sorted([face[i], face[i - 1]]))
                    output.add(edge)
            self._edges = FiniteSet(*output)
        return self._edges

    def rotate(self, perm):
        if False:
            for i in range(10):
                print('nop')
        '\n        Apply a permutation to the polyhedron *in place*. The permutation\n        may be given as a Permutation instance or an integer indicating\n        which permutation from pgroup of the Polyhedron should be\n        applied.\n\n        This is an operation that is analogous to rotation about\n        an axis by a fixed increment.\n\n        Notes\n        =====\n\n        When a Permutation is applied, no check is done to see if that\n        is a valid permutation for the Polyhedron. For example, a cube\n        could be given a permutation which effectively swaps only 2\n        vertices. A valid permutation (that rotates the object in a\n        physical way) will be obtained if one only uses\n        permutations from the ``pgroup`` of the Polyhedron. On the other\n        hand, allowing arbitrary rotations (applications of permutations)\n        gives a way to follow named elements rather than indices since\n        Polyhedron allows vertices to be named while Permutation works\n        only with indices.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Polyhedron, Permutation\n        >>> from sympy.combinatorics.polyhedron import cube\n        >>> cube = cube.copy()\n        >>> cube.corners\n        (0, 1, 2, 3, 4, 5, 6, 7)\n        >>> cube.rotate(0)\n        >>> cube.corners\n        (1, 2, 3, 0, 5, 6, 7, 4)\n\n        A non-physical "rotation" that is not prohibited by this method:\n\n        >>> cube.reset()\n        >>> cube.rotate(Permutation([[1, 2]], size=8))\n        >>> cube.corners\n        (0, 2, 1, 3, 4, 5, 6, 7)\n\n        Polyhedron can be used to follow elements of set that are\n        identified by letters instead of integers:\n\n        >>> shadow = h5 = Polyhedron(list(\'abcde\'))\n        >>> p = Permutation([3, 0, 1, 2, 4])\n        >>> h5.rotate(p)\n        >>> h5.corners\n        (d, a, b, c, e)\n        >>> _ == shadow.corners\n        True\n        >>> copy = h5.copy()\n        >>> h5.rotate(p)\n        >>> h5.corners == copy.corners\n        False\n        '
        if not isinstance(perm, Perm):
            perm = self.pgroup[perm]
        elif perm.size != self.size:
            raise ValueError('Polyhedron and Permutation sizes differ.')
        a = perm.array_form
        corners = [self.corners[a[i]] for i in range(len(self.corners))]
        self._corners = tuple(corners)

    def reset(self):
        if False:
            i = 10
            return i + 15
        'Return corners to their original positions.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.polyhedron import tetrahedron as T\n        >>> T = T.copy()\n        >>> T.corners\n        (0, 1, 2, 3)\n        >>> T.rotate(0)\n        >>> T.corners\n        (0, 2, 3, 1)\n        >>> T.reset()\n        >>> T.corners\n        (0, 1, 2, 3)\n        '
        self._corners = self.args[0]

def _pgroup_calcs():
    if False:
        for i in range(10):
            print('nop')
    'Return the permutation groups for each of the polyhedra and the face\n    definitions: tetrahedron, cube, octahedron, dodecahedron, icosahedron,\n    tetrahedron_faces, cube_faces, octahedron_faces, dodecahedron_faces,\n    icosahedron_faces\n\n    Explanation\n    ===========\n\n    (This author did not find and did not know of a better way to do it though\n    there likely is such a way.)\n\n    Although only 2 permutations are needed for a polyhedron in order to\n    generate all the possible orientations, a group of permutations is\n    provided instead. A set of permutations is called a "group" if::\n\n    a*b = c (for any pair of permutations in the group, a and b, their\n    product, c, is in the group)\n\n    a*(b*c) = (a*b)*c (for any 3 permutations in the group associativity holds)\n\n    there is an identity permutation, I, such that I*a = a*I for all elements\n    in the group\n\n    a*b = I (the inverse of each permutation is also in the group)\n\n    None of the polyhedron groups defined follow these definitions of a group.\n    Instead, they are selected to contain those permutations whose powers\n    alone will construct all orientations of the polyhedron, i.e. for\n    permutations ``a``, ``b``, etc... in the group, ``a, a**2, ..., a**o_a``,\n    ``b, b**2, ..., b**o_b``, etc... (where ``o_i`` is the order of\n    permutation ``i``) generate all permutations of the polyhedron instead of\n    mixed products like ``a*b``, ``a*b**2``, etc....\n\n    Note that for a polyhedron with n vertices, the valid permutations of the\n    vertices exclude those that do not maintain its faces. e.g. the\n    permutation BCDE of a square\'s four corners, ABCD, is a valid\n    permutation while CBDE is not (because this would twist the square).\n\n    Examples\n    ========\n\n    The is_group checks for: closure, the presence of the Identity permutation,\n    and the presence of the inverse for each of the elements in the group. This\n    confirms that none of the polyhedra are true groups:\n\n    >>> from sympy.combinatorics.polyhedron import (\n    ... tetrahedron, cube, octahedron, dodecahedron, icosahedron)\n    ...\n    >>> polyhedra = (tetrahedron, cube, octahedron, dodecahedron, icosahedron)\n    >>> [h.pgroup.is_group for h in polyhedra]\n    ...\n    [True, True, True, True, True]\n\n    Although tests in polyhedron\'s test suite check that powers of the\n    permutations in the groups generate all permutations of the vertices\n    of the polyhedron, here we also demonstrate the powers of the given\n    permutations create a complete group for the tetrahedron:\n\n    >>> from sympy.combinatorics import Permutation, PermutationGroup\n    >>> for h in polyhedra[:1]:\n    ...     G = h.pgroup\n    ...     perms = set()\n    ...     for g in G:\n    ...         for e in range(g.order()):\n    ...             p = tuple((g**e).array_form)\n    ...             perms.add(p)\n    ...\n    ...     perms = [Permutation(p) for p in perms]\n    ...     assert PermutationGroup(perms).is_group\n\n    In addition to doing the above, the tests in the suite confirm that the\n    faces are all present after the application of each permutation.\n\n    References\n    ==========\n\n    .. [1] https://dogschool.tripod.com/trianglegroup.html\n\n    '

    def _pgroup_of_double(polyh, ordered_faces, pgroup):
        if False:
            i = 10
            return i + 15
        n = len(ordered_faces[0])
        fmap = dict(zip(ordered_faces, range(len(ordered_faces))))
        flat_faces = flatten(ordered_faces)
        new_pgroup = []
        for (i, p) in enumerate(pgroup):
            h = polyh.copy()
            h.rotate(p)
            c = h.corners
            reorder = unflatten([c[j] for j in flat_faces], n)
            reorder = [tuple(map(as_int, minlex(f, directed=False))) for f in reorder]
            new_pgroup.append(Perm([fmap[f] for f in reorder]))
        return new_pgroup
    tetrahedron_faces = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 2, 3)]
    _t_pgroup = [Perm([[1, 2, 3], [0]]), Perm([[0, 1, 2], [3]]), Perm([[0, 3, 2], [1]]), Perm([[0, 3, 1], [2]]), Perm([[0, 1], [2, 3]]), Perm([[0, 2], [1, 3]]), Perm([[0, 3], [1, 2]])]
    tetrahedron = Polyhedron(range(4), tetrahedron_faces, _t_pgroup)
    cube_faces = [(0, 1, 2, 3), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (0, 3, 7, 4), (4, 5, 6, 7)]
    _c_pgroup = [Perm(p) for p in [[1, 2, 3, 0, 5, 6, 7, 4], [4, 0, 3, 7, 5, 1, 2, 6], [4, 5, 1, 0, 7, 6, 2, 3], [1, 0, 4, 5, 2, 3, 7, 6], [6, 2, 1, 5, 7, 3, 0, 4], [6, 7, 3, 2, 5, 4, 0, 1], [3, 7, 4, 0, 2, 6, 5, 1], [4, 7, 6, 5, 0, 3, 2, 1], [6, 5, 4, 7, 2, 1, 0, 3], [0, 3, 7, 4, 1, 2, 6, 5], [5, 1, 0, 4, 6, 2, 3, 7], [5, 6, 2, 1, 4, 7, 3, 0], [7, 4, 0, 3, 6, 5, 1, 2]]]
    cube = Polyhedron(range(8), cube_faces, _c_pgroup)
    octahedron_faces = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 1, 4), (1, 2, 5), (2, 3, 5), (3, 4, 5), (1, 4, 5)]
    octahedron = Polyhedron(range(6), octahedron_faces, _pgroup_of_double(cube, cube_faces, _c_pgroup))
    dodecahedron_faces = [(0, 1, 2, 3, 4), (0, 1, 6, 10, 5), (1, 2, 7, 11, 6), (2, 3, 8, 12, 7), (3, 4, 9, 13, 8), (0, 4, 9, 14, 5), (5, 10, 16, 15, 14), (6, 10, 16, 17, 11), (7, 11, 17, 18, 12), (8, 12, 18, 19, 13), (9, 13, 19, 15, 14), (15, 16, 17, 18, 19)]

    def _string_to_perm(s):
        if False:
            return 10
        rv = [Perm(range(20))]
        p = None
        for si in s:
            if si not in '01':
                count = int(si) - 1
            else:
                count = 1
                if si == '0':
                    p = _f0
                elif si == '1':
                    p = _f1
            rv.extend([p] * count)
        return Perm.rmul(*rv)
    _f0 = Perm([1, 2, 3, 4, 0, 6, 7, 8, 9, 5, 11, 12, 13, 14, 10, 16, 17, 18, 19, 15])
    _f1 = Perm([5, 0, 4, 9, 14, 10, 1, 3, 13, 15, 6, 2, 8, 19, 16, 17, 11, 7, 12, 18])
    _dodeca_pgroup = [_f0, _f1] + [_string_to_perm(s) for s in '\n    0104 140 014 0410\n    010 1403 03104 04103 102\n    120 1304 01303 021302 03130\n    0412041 041204103 04120410 041204104 041204102\n    10 01 1402 0140 04102 0412 1204 1302 0130 03120'.strip().split()]
    dodecahedron = Polyhedron(range(20), dodecahedron_faces, _dodeca_pgroup)
    icosahedron_faces = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 1, 5), (1, 6, 7), (1, 2, 7), (2, 7, 8), (2, 3, 8), (3, 8, 9), (3, 4, 9), (4, 9, 10), (4, 5, 10), (5, 6, 10), (1, 5, 6), (6, 7, 11), (7, 8, 11), (8, 9, 11), (9, 10, 11), (6, 10, 11)]
    icosahedron = Polyhedron(range(12), icosahedron_faces, _pgroup_of_double(dodecahedron, dodecahedron_faces, _dodeca_pgroup))
    return (tetrahedron, cube, octahedron, dodecahedron, icosahedron, tetrahedron_faces, cube_faces, octahedron_faces, dodecahedron_faces, icosahedron_faces)
tetrahedron = Polyhedron(Tuple(0, 1, 2, 3), Tuple(Tuple(0, 1, 2), Tuple(0, 2, 3), Tuple(0, 1, 3), Tuple(1, 2, 3)), Tuple(Perm(1, 2, 3), Perm(3)(0, 1, 2), Perm(0, 3, 2), Perm(0, 3, 1), Perm(0, 1)(2, 3), Perm(0, 2)(1, 3), Perm(0, 3)(1, 2)))
cube = Polyhedron(Tuple(0, 1, 2, 3, 4, 5, 6, 7), Tuple(Tuple(0, 1, 2, 3), Tuple(0, 1, 5, 4), Tuple(1, 2, 6, 5), Tuple(2, 3, 7, 6), Tuple(0, 3, 7, 4), Tuple(4, 5, 6, 7)), Tuple(Perm(0, 1, 2, 3)(4, 5, 6, 7), Perm(0, 4, 5, 1)(2, 3, 7, 6), Perm(0, 4, 7, 3)(1, 5, 6, 2), Perm(0, 1)(2, 4)(3, 5)(6, 7), Perm(0, 6)(1, 2)(3, 5)(4, 7), Perm(0, 6)(1, 7)(2, 3)(4, 5), Perm(0, 3)(1, 7)(2, 4)(5, 6), Perm(0, 4)(1, 7)(2, 6)(3, 5), Perm(0, 6)(1, 5)(2, 4)(3, 7), Perm(1, 3, 4)(2, 7, 5), Perm(7)(0, 5, 2)(3, 4, 6), Perm(0, 5, 7)(1, 6, 3), Perm(0, 7, 2)(1, 4, 6)))
octahedron = Polyhedron(Tuple(0, 1, 2, 3, 4, 5), Tuple(Tuple(0, 1, 2), Tuple(0, 2, 3), Tuple(0, 3, 4), Tuple(0, 1, 4), Tuple(1, 2, 5), Tuple(2, 3, 5), Tuple(3, 4, 5), Tuple(1, 4, 5)), Tuple(Perm(5)(1, 2, 3, 4), Perm(0, 4, 5, 2), Perm(0, 1, 5, 3), Perm(0, 1)(2, 4)(3, 5), Perm(0, 2)(1, 3)(4, 5), Perm(0, 3)(1, 5)(2, 4), Perm(0, 4)(1, 3)(2, 5), Perm(0, 5)(1, 4)(2, 3), Perm(0, 5)(1, 2)(3, 4), Perm(0, 4, 1)(2, 3, 5), Perm(0, 1, 2)(3, 4, 5), Perm(0, 2, 3)(1, 5, 4), Perm(0, 4, 3)(1, 5, 2)))
dodecahedron = Polyhedron(Tuple(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19), Tuple(Tuple(0, 1, 2, 3, 4), Tuple(0, 1, 6, 10, 5), Tuple(1, 2, 7, 11, 6), Tuple(2, 3, 8, 12, 7), Tuple(3, 4, 9, 13, 8), Tuple(0, 4, 9, 14, 5), Tuple(5, 10, 16, 15, 14), Tuple(6, 10, 16, 17, 11), Tuple(7, 11, 17, 18, 12), Tuple(8, 12, 18, 19, 13), Tuple(9, 13, 19, 15, 14), Tuple(15, 16, 17, 18, 19)), Tuple(Perm(0, 1, 2, 3, 4)(5, 6, 7, 8, 9)(10, 11, 12, 13, 14)(15, 16, 17, 18, 19), Perm(0, 5, 10, 6, 1)(2, 4, 14, 16, 11)(3, 9, 15, 17, 7)(8, 13, 19, 18, 12), Perm(0, 10, 17, 12, 3)(1, 6, 11, 7, 2)(4, 5, 16, 18, 8)(9, 14, 15, 19, 13), Perm(0, 6, 17, 19, 9)(1, 11, 18, 13, 4)(2, 7, 12, 8, 3)(5, 10, 16, 15, 14), Perm(0, 2, 12, 19, 14)(1, 7, 18, 15, 5)(3, 8, 13, 9, 4)(6, 11, 17, 16, 10), Perm(0, 4, 9, 14, 5)(1, 3, 13, 15, 10)(2, 8, 19, 16, 6)(7, 12, 18, 17, 11), Perm(0, 1)(2, 5)(3, 10)(4, 6)(7, 14)(8, 16)(9, 11)(12, 15)(13, 17)(18, 19), Perm(0, 7)(1, 2)(3, 6)(4, 11)(5, 12)(8, 10)(9, 17)(13, 16)(14, 18)(15, 19), Perm(0, 12)(1, 8)(2, 3)(4, 7)(5, 18)(6, 13)(9, 11)(10, 19)(14, 17)(15, 16), Perm(0, 8)(1, 13)(2, 9)(3, 4)(5, 12)(6, 19)(7, 14)(10, 18)(11, 15)(16, 17), Perm(0, 4)(1, 9)(2, 14)(3, 5)(6, 13)(7, 15)(8, 10)(11, 19)(12, 16)(17, 18), Perm(0, 5)(1, 14)(2, 15)(3, 16)(4, 10)(6, 9)(7, 19)(8, 17)(11, 13)(12, 18), Perm(0, 11)(1, 6)(2, 10)(3, 16)(4, 17)(5, 7)(8, 15)(9, 18)(12, 14)(13, 19), Perm(0, 18)(1, 12)(2, 7)(3, 11)(4, 17)(5, 19)(6, 8)(9, 16)(10, 13)(14, 15), Perm(0, 18)(1, 19)(2, 13)(3, 8)(4, 12)(5, 17)(6, 15)(7, 9)(10, 16)(11, 14), Perm(0, 13)(1, 19)(2, 15)(3, 14)(4, 9)(5, 8)(6, 18)(7, 16)(10, 12)(11, 17), Perm(0, 16)(1, 15)(2, 19)(3, 18)(4, 17)(5, 10)(6, 14)(7, 13)(8, 12)(9, 11), Perm(0, 18)(1, 17)(2, 16)(3, 15)(4, 19)(5, 12)(6, 11)(7, 10)(8, 14)(9, 13), Perm(0, 15)(1, 19)(2, 18)(3, 17)(4, 16)(5, 14)(6, 13)(7, 12)(8, 11)(9, 10), Perm(0, 17)(1, 16)(2, 15)(3, 19)(4, 18)(5, 11)(6, 10)(7, 14)(8, 13)(9, 12), Perm(0, 19)(1, 18)(2, 17)(3, 16)(4, 15)(5, 13)(6, 12)(7, 11)(8, 10)(9, 14), Perm(1, 4, 5)(2, 9, 10)(3, 14, 6)(7, 13, 16)(8, 15, 11)(12, 19, 17), Perm(19)(0, 6, 2)(3, 5, 11)(4, 10, 7)(8, 14, 17)(9, 16, 12)(13, 15, 18), Perm(0, 11, 8)(1, 7, 3)(4, 6, 12)(5, 17, 13)(9, 10, 18)(14, 16, 19), Perm(0, 7, 13)(1, 12, 9)(2, 8, 4)(5, 11, 19)(6, 18, 14)(10, 17, 15), Perm(0, 3, 9)(1, 8, 14)(2, 13, 5)(6, 12, 15)(7, 19, 10)(11, 18, 16), Perm(0, 14, 10)(1, 9, 16)(2, 13, 17)(3, 19, 11)(4, 15, 6)(7, 8, 18), Perm(0, 16, 7)(1, 10, 11)(2, 5, 17)(3, 14, 18)(4, 15, 12)(8, 9, 19), Perm(0, 16, 13)(1, 17, 8)(2, 11, 12)(3, 6, 18)(4, 10, 19)(5, 15, 9), Perm(0, 11, 15)(1, 17, 14)(2, 18, 9)(3, 12, 13)(4, 7, 19)(5, 6, 16), Perm(0, 8, 15)(1, 12, 16)(2, 18, 10)(3, 19, 5)(4, 13, 14)(6, 7, 17)))
icosahedron = Polyhedron(Tuple(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), Tuple(Tuple(0, 1, 2), Tuple(0, 2, 3), Tuple(0, 3, 4), Tuple(0, 4, 5), Tuple(0, 1, 5), Tuple(1, 6, 7), Tuple(1, 2, 7), Tuple(2, 7, 8), Tuple(2, 3, 8), Tuple(3, 8, 9), Tuple(3, 4, 9), Tuple(4, 9, 10), Tuple(4, 5, 10), Tuple(5, 6, 10), Tuple(1, 5, 6), Tuple(6, 7, 11), Tuple(7, 8, 11), Tuple(8, 9, 11), Tuple(9, 10, 11), Tuple(6, 10, 11)), Tuple(Perm(11)(1, 2, 3, 4, 5)(6, 7, 8, 9, 10), Perm(0, 5, 6, 7, 2)(3, 4, 10, 11, 8), Perm(0, 1, 7, 8, 3)(4, 5, 6, 11, 9), Perm(0, 2, 8, 9, 4)(1, 7, 11, 10, 5), Perm(0, 3, 9, 10, 5)(1, 2, 8, 11, 6), Perm(0, 4, 10, 6, 1)(2, 3, 9, 11, 7), Perm(0, 1)(2, 5)(3, 6)(4, 7)(8, 10)(9, 11), Perm(0, 2)(1, 3)(4, 7)(5, 8)(6, 9)(10, 11), Perm(0, 3)(1, 9)(2, 4)(5, 8)(6, 11)(7, 10), Perm(0, 4)(1, 9)(2, 10)(3, 5)(6, 8)(7, 11), Perm(0, 5)(1, 4)(2, 10)(3, 6)(7, 9)(8, 11), Perm(0, 6)(1, 5)(2, 10)(3, 11)(4, 7)(8, 9), Perm(0, 7)(1, 2)(3, 6)(4, 11)(5, 8)(9, 10), Perm(0, 8)(1, 9)(2, 3)(4, 7)(5, 11)(6, 10), Perm(0, 9)(1, 11)(2, 10)(3, 4)(5, 8)(6, 7), Perm(0, 10)(1, 9)(2, 11)(3, 6)(4, 5)(7, 8), Perm(0, 11)(1, 6)(2, 10)(3, 9)(4, 8)(5, 7), Perm(0, 11)(1, 8)(2, 7)(3, 6)(4, 10)(5, 9), Perm(0, 11)(1, 10)(2, 9)(3, 8)(4, 7)(5, 6), Perm(0, 11)(1, 7)(2, 6)(3, 10)(4, 9)(5, 8), Perm(0, 11)(1, 9)(2, 8)(3, 7)(4, 6)(5, 10), Perm(0, 5, 1)(2, 4, 6)(3, 10, 7)(8, 9, 11), Perm(0, 1, 2)(3, 5, 7)(4, 6, 8)(9, 10, 11), Perm(0, 2, 3)(1, 8, 4)(5, 7, 9)(6, 11, 10), Perm(0, 3, 4)(1, 8, 10)(2, 9, 5)(6, 7, 11), Perm(0, 4, 5)(1, 3, 10)(2, 9, 6)(7, 8, 11), Perm(0, 10, 7)(1, 5, 6)(2, 4, 11)(3, 9, 8), Perm(0, 6, 8)(1, 7, 2)(3, 5, 11)(4, 10, 9), Perm(0, 7, 9)(1, 11, 4)(2, 8, 3)(5, 6, 10), Perm(0, 8, 10)(1, 7, 6)(2, 11, 5)(3, 9, 4), Perm(0, 9, 6)(1, 3, 11)(2, 8, 7)(4, 10, 5)))
tetrahedron_faces = [tuple(arg) for arg in tetrahedron.faces]
cube_faces = [tuple(arg) for arg in cube.faces]
octahedron_faces = [tuple(arg) for arg in octahedron.faces]
dodecahedron_faces = [tuple(arg) for arg in dodecahedron.faces]
icosahedron_faces = [tuple(arg) for arg in icosahedron.faces]