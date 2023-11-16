from sympy import diff, expand, sin, cos, sympify, eye, zeros, ImmutableMatrix as Matrix, MatrixBase
from sympy.core.symbol import Symbol
from sympy.simplify.trigsimp import trigsimp
from sympy.physics.vector.vector import Vector, _check_vector
from sympy.utilities.misc import translate
from warnings import warn
__all__ = ['CoordinateSym', 'ReferenceFrame']

class CoordinateSym(Symbol):
    """
    A coordinate symbol/base scalar associated wrt a Reference Frame.

    Ideally, users should not instantiate this class. Instances of
    this class must only be accessed through the corresponding frame
    as 'frame[index]'.

    CoordinateSyms having the same frame and index parameters are equal
    (even though they may be instantiated separately).

    Parameters
    ==========

    name : string
        The display name of the CoordinateSym

    frame : ReferenceFrame
        The reference frame this base scalar belongs to

    index : 0, 1 or 2
        The index of the dimension denoted by this coordinate variable

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame, CoordinateSym
    >>> A = ReferenceFrame('A')
    >>> A[1]
    A_y
    >>> type(A[0])
    <class 'sympy.physics.vector.frame.CoordinateSym'>
    >>> a_y = CoordinateSym('a_y', A, 1)
    >>> a_y == A[1]
    True

    """

    def __new__(cls, name, frame, index):
        if False:
            return 10
        assumptions = {}
        super()._sanitize(assumptions, cls)
        obj = super().__xnew__(cls, name, **assumptions)
        _check_frame(frame)
        if index not in range(0, 3):
            raise ValueError('Invalid index specified')
        obj._id = (frame, index)
        return obj

    def __getnewargs_ex__(self):
        if False:
            i = 10
            return i + 15
        return ((self.name, *self._id), {})

    @property
    def frame(self):
        if False:
            for i in range(10):
                print('nop')
        return self._id[0]

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, CoordinateSym):
            if other._id == self._id:
                return True
        return False

    def __ne__(self, other):
        if False:
            while True:
                i = 10
        return not self == other

    def __hash__(self):
        if False:
            while True:
                i = 10
        return (self._id[0].__hash__(), self._id[1]).__hash__()

class ReferenceFrame:
    """A reference frame in classical mechanics.

    ReferenceFrame is a class used to represent a reference frame in classical
    mechanics. It has a standard basis of three unit vectors in the frame's
    x, y, and z directions.

    It also can have a rotation relative to a parent frame; this rotation is
    defined by a direction cosine matrix relating this frame's basis vectors to
    the parent frame's basis vectors.  It can also have an angular velocity
    vector, defined in another frame.

    """
    _count = 0

    def __init__(self, name, indices=None, latexs=None, variables=None):
        if False:
            while True:
                i = 10
        "ReferenceFrame initialization method.\n\n        A ReferenceFrame has a set of orthonormal basis vectors, along with\n        orientations relative to other ReferenceFrames and angular velocities\n        relative to other ReferenceFrames.\n\n        Parameters\n        ==========\n\n        indices : tuple of str\n            Enables the reference frame's basis unit vectors to be accessed by\n            Python's square bracket indexing notation using the provided three\n            indice strings and alters the printing of the unit vectors to\n            reflect this choice.\n        latexs : tuple of str\n            Alters the LaTeX printing of the reference frame's basis unit\n            vectors to the provided three valid LaTeX strings.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.vector import ReferenceFrame, vlatex\n        >>> N = ReferenceFrame('N')\n        >>> N.x\n        N.x\n        >>> O = ReferenceFrame('O', indices=('1', '2', '3'))\n        >>> O.x\n        O['1']\n        >>> O['1']\n        O['1']\n        >>> P = ReferenceFrame('P', latexs=('A1', 'A2', 'A3'))\n        >>> vlatex(P.x)\n        'A1'\n\n        ``symbols()`` can be used to create multiple Reference Frames in one\n        step, for example:\n\n        >>> from sympy.physics.vector import ReferenceFrame\n        >>> from sympy import symbols\n        >>> A, B, C = symbols('A B C', cls=ReferenceFrame)\n        >>> D, E = symbols('D E', cls=ReferenceFrame, indices=('1', '2', '3'))\n        >>> A[0]\n        A_x\n        >>> D.x\n        D['1']\n        >>> E.y\n        E['2']\n        >>> type(A) == type(D)\n        True\n\n        Unit dyads for the ReferenceFrame can be accessed through the attributes ``xx``, ``xy``, etc. For example:\n\n        >>> from sympy.physics.vector import ReferenceFrame\n        >>> N = ReferenceFrame('N')\n        >>> N.yz\n        (N.y|N.z)\n        >>> N.zx\n        (N.z|N.x)\n        >>> P = ReferenceFrame('P', indices=['1', '2', '3'])\n        >>> P.xx\n        (P['1']|P['1'])\n        >>> P.zy\n        (P['3']|P['2'])\n\n        Unit dyadic is also accessible via the ``u`` attribute:\n\n        >>> from sympy.physics.vector import ReferenceFrame\n        >>> N = ReferenceFrame('N')\n        >>> N.u\n        (N.x|N.x) + (N.y|N.y) + (N.z|N.z)\n        >>> P = ReferenceFrame('P', indices=['1', '2', '3'])\n        >>> P.u\n        (P['1']|P['1']) + (P['2']|P['2']) + (P['3']|P['3'])\n\n        "
        if not isinstance(name, str):
            raise TypeError('Need to supply a valid name')
        if indices is not None:
            if not isinstance(indices, (tuple, list)):
                raise TypeError('Supply the indices as a list')
            if len(indices) != 3:
                raise ValueError('Supply 3 indices')
            for i in indices:
                if not isinstance(i, str):
                    raise TypeError('Indices must be strings')
            self.str_vecs = [name + "['" + indices[0] + "']", name + "['" + indices[1] + "']", name + "['" + indices[2] + "']"]
            self.pretty_vecs = [name.lower() + '_' + indices[0], name.lower() + '_' + indices[1], name.lower() + '_' + indices[2]]
            self.latex_vecs = ['\\mathbf{\\hat{%s}_{%s}}' % (name.lower(), indices[0]), '\\mathbf{\\hat{%s}_{%s}}' % (name.lower(), indices[1]), '\\mathbf{\\hat{%s}_{%s}}' % (name.lower(), indices[2])]
            self.indices = indices
        else:
            self.str_vecs = [name + '.x', name + '.y', name + '.z']
            self.pretty_vecs = [name.lower() + '_x', name.lower() + '_y', name.lower() + '_z']
            self.latex_vecs = ['\\mathbf{\\hat{%s}_x}' % name.lower(), '\\mathbf{\\hat{%s}_y}' % name.lower(), '\\mathbf{\\hat{%s}_z}' % name.lower()]
            self.indices = ['x', 'y', 'z']
        if latexs is not None:
            if not isinstance(latexs, (tuple, list)):
                raise TypeError('Supply the indices as a list')
            if len(latexs) != 3:
                raise ValueError('Supply 3 indices')
            for i in latexs:
                if not isinstance(i, str):
                    raise TypeError('Latex entries must be strings')
            self.latex_vecs = latexs
        self.name = name
        self._var_dict = {}
        self._dcm_dict = {}
        self._dcm_cache = {}
        self._ang_vel_dict = {}
        self._ang_acc_dict = {}
        self._dlist = [self._dcm_dict, self._ang_vel_dict, self._ang_acc_dict]
        self._cur = 0
        self._x = Vector([(Matrix([1, 0, 0]), self)])
        self._y = Vector([(Matrix([0, 1, 0]), self)])
        self._z = Vector([(Matrix([0, 0, 1]), self)])
        if variables is not None:
            if not isinstance(variables, (tuple, list)):
                raise TypeError('Supply the variable names as a list/tuple')
            if len(variables) != 3:
                raise ValueError('Supply 3 variable names')
            for i in variables:
                if not isinstance(i, str):
                    raise TypeError('Variable names must be strings')
        else:
            variables = [name + '_x', name + '_y', name + '_z']
        self.varlist = (CoordinateSym(variables[0], self, 0), CoordinateSym(variables[1], self, 1), CoordinateSym(variables[2], self, 2))
        ReferenceFrame._count += 1
        self.index = ReferenceFrame._count

    def __getitem__(self, ind):
        if False:
            print('Hello World!')
        '\n        Returns basis vector for the provided index, if the index is a string.\n\n        If the index is a number, returns the coordinate variable correspon-\n        -ding to that index.\n        '
        if not isinstance(ind, str):
            if ind < 3:
                return self.varlist[ind]
            else:
                raise ValueError('Invalid index provided')
        if self.indices[0] == ind:
            return self.x
        if self.indices[1] == ind:
            return self.y
        if self.indices[2] == ind:
            return self.z
        else:
            raise ValueError('Not a defined index')

    def __iter__(self):
        if False:
            return 10
        return iter([self.x, self.y, self.z])

    def __str__(self):
        if False:
            return 10
        'Returns the name of the frame. '
        return self.name
    __repr__ = __str__

    def _dict_list(self, other, num):
        if False:
            return 10
        "Returns an inclusive list of reference frames that connect this\n        reference frame to the provided reference frame.\n\n        Parameters\n        ==========\n        other : ReferenceFrame\n            The other reference frame to look for a connecting relationship to.\n        num : integer\n            ``0``, ``1``, and ``2`` will look for orientation, angular\n            velocity, and angular acceleration relationships between the two\n            frames, respectively.\n\n        Returns\n        =======\n        list\n            Inclusive list of reference frames that connect this reference\n            frame to the other reference frame.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.vector import ReferenceFrame\n        >>> A = ReferenceFrame('A')\n        >>> B = ReferenceFrame('B')\n        >>> C = ReferenceFrame('C')\n        >>> D = ReferenceFrame('D')\n        >>> B.orient_axis(A, A.x, 1.0)\n        >>> C.orient_axis(B, B.x, 1.0)\n        >>> D.orient_axis(C, C.x, 1.0)\n        >>> D._dict_list(A, 0)\n        [D, C, B, A]\n\n        Raises\n        ======\n\n        ValueError\n            When no path is found between the two reference frames or ``num``\n            is an incorrect value.\n\n        "
        connect_type = {0: 'orientation', 1: 'angular velocity', 2: 'angular acceleration'}
        if num not in connect_type.keys():
            raise ValueError('Valid values for num are 0, 1, or 2.')
        possible_connecting_paths = [[self]]
        oldlist = [[]]
        while possible_connecting_paths != oldlist:
            oldlist = possible_connecting_paths[:]
            for frame_list in possible_connecting_paths:
                frames_adjacent_to_last = frame_list[-1]._dlist[num].keys()
                for adjacent_frame in frames_adjacent_to_last:
                    if adjacent_frame not in frame_list:
                        connecting_path = frame_list + [adjacent_frame]
                        if connecting_path not in possible_connecting_paths:
                            possible_connecting_paths.append(connecting_path)
        for connecting_path in oldlist:
            if connecting_path[-1] != other:
                possible_connecting_paths.remove(connecting_path)
        possible_connecting_paths.sort(key=len)
        if len(possible_connecting_paths) != 0:
            return possible_connecting_paths[0]
        msg = 'No connecting {} path found between {} and {}.'
        raise ValueError(msg.format(connect_type[num], self.name, other.name))

    def _w_diff_dcm(self, otherframe):
        if False:
            i = 10
            return i + 15
        'Angular velocity from time differentiating the DCM. '
        from sympy.physics.vector.functions import dynamicsymbols
        dcm2diff = otherframe.dcm(self)
        diffed = dcm2diff.diff(dynamicsymbols._t)
        angvelmat = diffed * dcm2diff.T
        w1 = trigsimp(expand(angvelmat[7]), recursive=True)
        w2 = trigsimp(expand(angvelmat[2]), recursive=True)
        w3 = trigsimp(expand(angvelmat[3]), recursive=True)
        return Vector([(Matrix([w1, w2, w3]), otherframe)])

    def variable_map(self, otherframe):
        if False:
            i = 10
            return i + 15
        "\n        Returns a dictionary which expresses the coordinate variables\n        of this frame in terms of the variables of otherframe.\n\n        If Vector.simp is True, returns a simplified version of the mapped\n        values. Else, returns them without simplification.\n\n        Simplification of the expressions may take time.\n\n        Parameters\n        ==========\n\n        otherframe : ReferenceFrame\n            The other frame to map the variables to\n\n        Examples\n        ========\n\n        >>> from sympy.physics.vector import ReferenceFrame, dynamicsymbols\n        >>> A = ReferenceFrame('A')\n        >>> q = dynamicsymbols('q')\n        >>> B = A.orientnew('B', 'Axis', [q, A.z])\n        >>> A.variable_map(B)\n        {A_x: B_x*cos(q(t)) - B_y*sin(q(t)), A_y: B_x*sin(q(t)) + B_y*cos(q(t)), A_z: B_z}\n\n        "
        _check_frame(otherframe)
        if (otherframe, Vector.simp) in self._var_dict:
            return self._var_dict[otherframe, Vector.simp]
        else:
            vars_matrix = self.dcm(otherframe) * Matrix(otherframe.varlist)
            mapping = {}
            for (i, x) in enumerate(self):
                if Vector.simp:
                    mapping[self.varlist[i]] = trigsimp(vars_matrix[i], method='fu')
                else:
                    mapping[self.varlist[i]] = vars_matrix[i]
            self._var_dict[otherframe, Vector.simp] = mapping
            return mapping

    def ang_acc_in(self, otherframe):
        if False:
            print('Hello World!')
        "Returns the angular acceleration Vector of the ReferenceFrame.\n\n        Effectively returns the Vector:\n\n        ``N_alpha_B``\n\n        which represent the angular acceleration of B in N, where B is self,\n        and N is otherframe.\n\n        Parameters\n        ==========\n\n        otherframe : ReferenceFrame\n            The ReferenceFrame which the angular acceleration is returned in.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.vector import ReferenceFrame\n        >>> N = ReferenceFrame('N')\n        >>> A = ReferenceFrame('A')\n        >>> V = 10 * N.x\n        >>> A.set_ang_acc(N, V)\n        >>> A.ang_acc_in(N)\n        10*N.x\n\n        "
        _check_frame(otherframe)
        if otherframe in self._ang_acc_dict:
            return self._ang_acc_dict[otherframe]
        else:
            return self.ang_vel_in(otherframe).dt(otherframe)

    def ang_vel_in(self, otherframe):
        if False:
            i = 10
            return i + 15
        "Returns the angular velocity Vector of the ReferenceFrame.\n\n        Effectively returns the Vector:\n\n        ^N omega ^B\n\n        which represent the angular velocity of B in N, where B is self, and\n        N is otherframe.\n\n        Parameters\n        ==========\n\n        otherframe : ReferenceFrame\n            The ReferenceFrame which the angular velocity is returned in.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.vector import ReferenceFrame\n        >>> N = ReferenceFrame('N')\n        >>> A = ReferenceFrame('A')\n        >>> V = 10 * N.x\n        >>> A.set_ang_vel(N, V)\n        >>> A.ang_vel_in(N)\n        10*N.x\n\n        "
        _check_frame(otherframe)
        flist = self._dict_list(otherframe, 1)
        outvec = Vector(0)
        for i in range(len(flist) - 1):
            outvec += flist[i]._ang_vel_dict[flist[i + 1]]
        return outvec

    def dcm(self, otherframe):
        if False:
            return 10
        'Returns the direction cosine matrix of this reference frame\n        relative to the provided reference frame.\n\n        The returned matrix can be used to express the orthogonal unit vectors\n        of this frame in terms of the orthogonal unit vectors of\n        ``otherframe``.\n\n        Parameters\n        ==========\n\n        otherframe : ReferenceFrame\n            The reference frame which the direction cosine matrix of this frame\n            is formed relative to.\n\n        Examples\n        ========\n\n        The following example rotates the reference frame A relative to N by a\n        simple rotation and then calculates the direction cosine matrix of N\n        relative to A.\n\n        >>> from sympy import symbols, sin, cos\n        >>> from sympy.physics.vector import ReferenceFrame\n        >>> q1 = symbols(\'q1\')\n        >>> N = ReferenceFrame(\'N\')\n        >>> A = ReferenceFrame(\'A\')\n        >>> A.orient_axis(N, q1, N.x)\n        >>> N.dcm(A)\n        Matrix([\n        [1,       0,        0],\n        [0, cos(q1), -sin(q1)],\n        [0, sin(q1),  cos(q1)]])\n\n        The second row of the above direction cosine matrix represents the\n        ``N.y`` unit vector in N expressed in A. Like so:\n\n        >>> Ny = 0*A.x + cos(q1)*A.y - sin(q1)*A.z\n\n        Thus, expressing ``N.y`` in A should return the same result:\n\n        >>> N.y.express(A)\n        cos(q1)*A.y - sin(q1)*A.z\n\n        Notes\n        =====\n\n        It is important to know what form of the direction cosine matrix is\n        returned. If ``B.dcm(A)`` is called, it means the "direction cosine\n        matrix of B rotated relative to A". This is the matrix\n        :math:`{}^B\\mathbf{C}^A` shown in the following relationship:\n\n        .. math::\n\n           \\begin{bmatrix}\n             \\hat{\\mathbf{b}}_1 \\\\\n             \\hat{\\mathbf{b}}_2 \\\\\n             \\hat{\\mathbf{b}}_3\n           \\end{bmatrix}\n           =\n           {}^B\\mathbf{C}^A\n           \\begin{bmatrix}\n             \\hat{\\mathbf{a}}_1 \\\\\n             \\hat{\\mathbf{a}}_2 \\\\\n             \\hat{\\mathbf{a}}_3\n           \\end{bmatrix}.\n\n        :math:`{}^B\\mathbf{C}^A` is the matrix that expresses the B unit\n        vectors in terms of the A unit vectors.\n\n        '
        _check_frame(otherframe)
        if otherframe in self._dcm_cache:
            return self._dcm_cache[otherframe]
        flist = self._dict_list(otherframe, 0)
        outdcm = eye(3)
        for i in range(len(flist) - 1):
            outdcm = outdcm * flist[i]._dcm_dict[flist[i + 1]]
        self._dcm_cache[otherframe] = outdcm
        otherframe._dcm_cache[self] = outdcm.T
        return outdcm

    def _dcm(self, parent, parent_orient):
        if False:
            while True:
                i = 10
        frames = self._dcm_cache.keys()
        dcm_dict_del = []
        dcm_cache_del = []
        if parent in frames:
            for frame in frames:
                if frame in self._dcm_dict:
                    dcm_dict_del += [frame]
                dcm_cache_del += [frame]
            for frame in dcm_dict_del:
                del frame._dcm_dict[self]
            for frame in dcm_cache_del:
                del frame._dcm_cache[self]
            self._dcm_dict = self._dlist[0] = {}
            self._dcm_cache = {}
        else:
            visited = []
            queue = list(frames)
            cont = True
            while queue and cont:
                node = queue.pop(0)
                if node not in visited:
                    visited.append(node)
                    neighbors = node._dcm_dict.keys()
                    for neighbor in neighbors:
                        if neighbor == parent:
                            warn('Loops are defined among the orientation of frames. This is likely not desired and may cause errors in your calculations.')
                            cont = False
                            break
                        queue.append(neighbor)
        self._dcm_dict.update({parent: parent_orient.T})
        parent._dcm_dict.update({self: parent_orient})
        self._dcm_cache.update({parent: parent_orient.T})
        parent._dcm_cache.update({self: parent_orient})

    def orient_axis(self, parent, axis, angle):
        if False:
            return 10
        "Sets the orientation of this reference frame with respect to a\n        parent reference frame by rotating through an angle about an axis fixed\n        in the parent reference frame.\n\n        Parameters\n        ==========\n\n        parent : ReferenceFrame\n            Reference frame that this reference frame will be rotated relative\n            to.\n        axis : Vector\n            Vector fixed in the parent frame about about which this frame is\n            rotated. It need not be a unit vector and the rotation follows the\n            right hand rule.\n        angle : sympifiable\n            Angle in radians by which it the frame is to be rotated.\n\n        Warns\n        ======\n\n        UserWarning\n            If the orientation creates a kinematic loop.\n\n        Examples\n        ========\n\n        Setup variables for the examples:\n\n        >>> from sympy import symbols\n        >>> from sympy.physics.vector import ReferenceFrame\n        >>> q1 = symbols('q1')\n        >>> N = ReferenceFrame('N')\n        >>> B = ReferenceFrame('B')\n        >>> B.orient_axis(N, N.x, q1)\n\n        The ``orient_axis()`` method generates a direction cosine matrix and\n        its transpose which defines the orientation of B relative to N and vice\n        versa. Once orient is called, ``dcm()`` outputs the appropriate\n        direction cosine matrix:\n\n        >>> B.dcm(N)\n        Matrix([\n        [1,       0,      0],\n        [0,  cos(q1), sin(q1)],\n        [0, -sin(q1), cos(q1)]])\n        >>> N.dcm(B)\n        Matrix([\n        [1,       0,        0],\n        [0, cos(q1), -sin(q1)],\n        [0, sin(q1),  cos(q1)]])\n\n        The following two lines show that the sense of the rotation can be\n        defined by negating the vector direction or the angle. Both lines\n        produce the same result.\n\n        >>> B.orient_axis(N, -N.x, q1)\n        >>> B.orient_axis(N, N.x, -q1)\n\n        "
        from sympy.physics.vector.functions import dynamicsymbols
        _check_frame(parent)
        if not isinstance(axis, Vector) and isinstance(angle, Vector):
            (axis, angle) = (angle, axis)
        axis = _check_vector(axis)
        theta = sympify(angle)
        if not axis.dt(parent) == 0:
            raise ValueError('Axis cannot be time-varying.')
        unit_axis = axis.express(parent).normalize()
        unit_col = unit_axis.args[0][0]
        parent_orient_axis = (eye(3) - unit_col * unit_col.T) * cos(theta) + Matrix([[0, -unit_col[2], unit_col[1]], [unit_col[2], 0, -unit_col[0]], [-unit_col[1], unit_col[0], 0]]) * sin(theta) + unit_col * unit_col.T
        self._dcm(parent, parent_orient_axis)
        thetad = theta.diff(dynamicsymbols._t)
        wvec = thetad * axis.express(parent).normalize()
        self._ang_vel_dict.update({parent: wvec})
        parent._ang_vel_dict.update({self: -wvec})
        self._var_dict = {}

    def orient_explicit(self, parent, dcm):
        if False:
            i = 10
            return i + 15
        "Sets the orientation of this reference frame relative to a parent\n        reference frame by explicitly setting the direction cosine matrix.\n\n        Parameters\n        ==========\n\n        parent : ReferenceFrame\n            Reference frame that this reference frame will be rotated relative\n            to.\n        dcm : Matrix, shape(3, 3)\n            Direction cosine matrix that specifies the relative rotation\n            between the two reference frames.\n\n        Warns\n        ======\n\n        UserWarning\n            If the orientation creates a kinematic loop.\n\n        Examples\n        ========\n\n        Setup variables for the examples:\n\n        >>> from sympy import symbols, Matrix, sin, cos\n        >>> from sympy.physics.vector import ReferenceFrame\n        >>> q1 = symbols('q1')\n        >>> A = ReferenceFrame('A')\n        >>> B = ReferenceFrame('B')\n        >>> N = ReferenceFrame('N')\n\n        A simple rotation of ``A`` relative to ``N`` about ``N.x`` is defined\n        by the following direction cosine matrix:\n\n        >>> dcm = Matrix([[1, 0, 0],\n        ...               [0, cos(q1), -sin(q1)],\n        ...               [0, sin(q1), cos(q1)]])\n        >>> A.orient_explicit(N, dcm)\n        >>> A.dcm(N)\n        Matrix([\n        [1,       0,      0],\n        [0,  cos(q1), sin(q1)],\n        [0, -sin(q1), cos(q1)]])\n\n        This is equivalent to using ``orient_axis()``:\n\n        >>> B.orient_axis(N, N.x, q1)\n        >>> B.dcm(N)\n        Matrix([\n        [1,       0,      0],\n        [0,  cos(q1), sin(q1)],\n        [0, -sin(q1), cos(q1)]])\n\n        **Note carefully that** ``N.dcm(B)`` **(the transpose) would be passed\n        into** ``orient_explicit()`` **for** ``A.dcm(N)`` **to match**\n        ``B.dcm(N)``:\n\n        >>> A.orient_explicit(N, N.dcm(B))\n        >>> A.dcm(N)\n        Matrix([\n        [1,       0,      0],\n        [0,  cos(q1), sin(q1)],\n        [0, -sin(q1), cos(q1)]])\n\n        "
        _check_frame(parent)
        if not isinstance(dcm, MatrixBase):
            raise TypeError('Amounts must be a SymPy Matrix type object.')
        parent_orient_dcm = dcm
        self._dcm(parent, parent_orient_dcm)
        wvec = self._w_diff_dcm(parent)
        self._ang_vel_dict.update({parent: wvec})
        parent._ang_vel_dict.update({self: -wvec})
        self._var_dict = {}

    def _rot(self, axis, angle):
        if False:
            return 10
        'DCM for simple axis 1,2,or 3 rotations.'
        if axis == 1:
            return Matrix([[1, 0, 0], [0, cos(angle), -sin(angle)], [0, sin(angle), cos(angle)]])
        elif axis == 2:
            return Matrix([[cos(angle), 0, sin(angle)], [0, 1, 0], [-sin(angle), 0, cos(angle)]])
        elif axis == 3:
            return Matrix([[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]])

    def _parse_consecutive_rotations(self, angles, rotation_order):
        if False:
            while True:
                i = 10
        "Helper for orient_body_fixed and orient_space_fixed.\n\n        Parameters\n        ==========\n        angles : 3-tuple of sympifiable\n            Three angles in radians used for the successive rotations.\n        rotation_order : 3 character string or 3 digit integer\n            Order of the rotations. The order can be specified by the strings\n            ``'XZX'``, ``'131'``, or the integer ``131``. There are 12 unique\n            valid rotation orders.\n\n        Returns\n        =======\n\n        amounts : list\n            List of sympifiables corresponding to the rotation angles.\n        rot_order : list\n            List of integers corresponding to the axis of rotation.\n        rot_matrices : list\n            List of DCM around the given axis with corresponding magnitude.\n\n        "
        amounts = list(angles)
        for (i, v) in enumerate(amounts):
            if not isinstance(v, Vector):
                amounts[i] = sympify(v)
        approved_orders = ('123', '231', '312', '132', '213', '321', '121', '131', '212', '232', '313', '323', '')
        rot_order = translate(str(rotation_order), 'XYZxyz', '123123')
        if rot_order not in approved_orders:
            raise TypeError('The rotation order is not a valid order.')
        rot_order = [int(r) for r in rot_order]
        if not len(amounts) == 3 & len(rot_order) == 3:
            raise TypeError('Body orientation takes 3 values & 3 orders')
        rot_matrices = [self._rot(order, amount) for (order, amount) in zip(rot_order, amounts)]
        return (amounts, rot_order, rot_matrices)

    def orient_body_fixed(self, parent, angles, rotation_order):
        if False:
            return 10
        'Rotates this reference frame relative to the parent reference frame\n        by right hand rotating through three successive body fixed simple axis\n        rotations. Each subsequent axis of rotation is about the "body fixed"\n        unit vectors of a new intermediate reference frame. This type of\n        rotation is also referred to rotating through the `Euler and Tait-Bryan\n        Angles`_.\n\n        .. _Euler and Tait-Bryan Angles: https://en.wikipedia.org/wiki/Euler_angles\n\n        The computed angular velocity in this method is by default expressed in\n        the child\'s frame, so it is most preferable to use ``u1 * child.x + u2 *\n        child.y + u3 * child.z`` as generalized speeds.\n\n        Parameters\n        ==========\n\n        parent : ReferenceFrame\n            Reference frame that this reference frame will be rotated relative\n            to.\n        angles : 3-tuple of sympifiable\n            Three angles in radians used for the successive rotations.\n        rotation_order : 3 character string or 3 digit integer\n            Order of the rotations about each intermediate reference frames\'\n            unit vectors. The Euler rotation about the X, Z\', X\'\' axes can be\n            specified by the strings ``\'XZX\'``, ``\'131\'``, or the integer\n            ``131``. There are 12 unique valid rotation orders (6 Euler and 6\n            Tait-Bryan): zxz, xyx, yzy, zyz, xzx, yxy, xyz, yzx, zxy, xzy, zyx,\n            and yxz.\n\n        Warns\n        ======\n\n        UserWarning\n            If the orientation creates a kinematic loop.\n\n        Examples\n        ========\n\n        Setup variables for the examples:\n\n        >>> from sympy import symbols\n        >>> from sympy.physics.vector import ReferenceFrame\n        >>> q1, q2, q3 = symbols(\'q1, q2, q3\')\n        >>> N = ReferenceFrame(\'N\')\n        >>> B = ReferenceFrame(\'B\')\n        >>> B1 = ReferenceFrame(\'B1\')\n        >>> B2 = ReferenceFrame(\'B2\')\n        >>> B3 = ReferenceFrame(\'B3\')\n\n        For example, a classic Euler Angle rotation can be done by:\n\n        >>> B.orient_body_fixed(N, (q1, q2, q3), \'XYX\')\n        >>> B.dcm(N)\n        Matrix([\n        [        cos(q2),                            sin(q1)*sin(q2),                           -sin(q2)*cos(q1)],\n        [sin(q2)*sin(q3), -sin(q1)*sin(q3)*cos(q2) + cos(q1)*cos(q3),  sin(q1)*cos(q3) + sin(q3)*cos(q1)*cos(q2)],\n        [sin(q2)*cos(q3), -sin(q1)*cos(q2)*cos(q3) - sin(q3)*cos(q1), -sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3)]])\n\n        This rotates reference frame B relative to reference frame N through\n        ``q1`` about ``N.x``, then rotates B again through ``q2`` about\n        ``B.y``, and finally through ``q3`` about ``B.x``. It is equivalent to\n        three successive ``orient_axis()`` calls:\n\n        >>> B1.orient_axis(N, N.x, q1)\n        >>> B2.orient_axis(B1, B1.y, q2)\n        >>> B3.orient_axis(B2, B2.x, q3)\n        >>> B3.dcm(N)\n        Matrix([\n        [        cos(q2),                            sin(q1)*sin(q2),                           -sin(q2)*cos(q1)],\n        [sin(q2)*sin(q3), -sin(q1)*sin(q3)*cos(q2) + cos(q1)*cos(q3),  sin(q1)*cos(q3) + sin(q3)*cos(q1)*cos(q2)],\n        [sin(q2)*cos(q3), -sin(q1)*cos(q2)*cos(q3) - sin(q3)*cos(q1), -sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3)]])\n\n        Acceptable rotation orders are of length 3, expressed in as a string\n        ``\'XYZ\'`` or ``\'123\'`` or integer ``123``. Rotations about an axis\n        twice in a row are prohibited.\n\n        >>> B.orient_body_fixed(N, (q1, q2, 0), \'ZXZ\')\n        >>> B.orient_body_fixed(N, (q1, q2, 0), \'121\')\n        >>> B.orient_body_fixed(N, (q1, q2, q3), 123)\n\n        '
        from sympy.physics.vector.functions import dynamicsymbols
        _check_frame(parent)
        (amounts, rot_order, rot_matrices) = self._parse_consecutive_rotations(angles, rotation_order)
        self._dcm(parent, rot_matrices[0] * rot_matrices[1] * rot_matrices[2])
        rot_vecs = [zeros(3, 1) for _ in range(3)]
        for (i, order) in enumerate(rot_order):
            rot_vecs[i][order - 1] = amounts[i].diff(dynamicsymbols._t)
        (u1, u2, u3) = rot_vecs[2] + rot_matrices[2].T * (rot_vecs[1] + rot_matrices[1].T * rot_vecs[0])
        wvec = u1 * self.x + u2 * self.y + u3 * self.z
        self._ang_vel_dict.update({parent: wvec})
        parent._ang_vel_dict.update({self: -wvec})
        self._var_dict = {}

    def orient_space_fixed(self, parent, angles, rotation_order):
        if False:
            return 10
        'Rotates this reference frame relative to the parent reference frame\n        by right hand rotating through three successive space fixed simple axis\n        rotations. Each subsequent axis of rotation is about the "space fixed"\n        unit vectors of the parent reference frame.\n\n        The computed angular velocity in this method is by default expressed in\n        the child\'s frame, so it is most preferable to use ``u1 * child.x + u2 *\n        child.y + u3 * child.z`` as generalized speeds.\n\n        Parameters\n        ==========\n        parent : ReferenceFrame\n            Reference frame that this reference frame will be rotated relative\n            to.\n        angles : 3-tuple of sympifiable\n            Three angles in radians used for the successive rotations.\n        rotation_order : 3 character string or 3 digit integer\n            Order of the rotations about the parent reference frame\'s unit\n            vectors. The order can be specified by the strings ``\'XZX\'``,\n            ``\'131\'``, or the integer ``131``. There are 12 unique valid\n            rotation orders.\n\n        Warns\n        ======\n\n        UserWarning\n            If the orientation creates a kinematic loop.\n\n        Examples\n        ========\n\n        Setup variables for the examples:\n\n        >>> from sympy import symbols\n        >>> from sympy.physics.vector import ReferenceFrame\n        >>> q1, q2, q3 = symbols(\'q1, q2, q3\')\n        >>> N = ReferenceFrame(\'N\')\n        >>> B = ReferenceFrame(\'B\')\n        >>> B1 = ReferenceFrame(\'B1\')\n        >>> B2 = ReferenceFrame(\'B2\')\n        >>> B3 = ReferenceFrame(\'B3\')\n\n        >>> B.orient_space_fixed(N, (q1, q2, q3), \'312\')\n        >>> B.dcm(N)\n        Matrix([\n        [ sin(q1)*sin(q2)*sin(q3) + cos(q1)*cos(q3), sin(q1)*cos(q2), sin(q1)*sin(q2)*cos(q3) - sin(q3)*cos(q1)],\n        [-sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1), cos(q1)*cos(q2), sin(q1)*sin(q3) + sin(q2)*cos(q1)*cos(q3)],\n        [                           sin(q3)*cos(q2),        -sin(q2),                           cos(q2)*cos(q3)]])\n\n        is equivalent to:\n\n        >>> B1.orient_axis(N, N.z, q1)\n        >>> B2.orient_axis(B1, N.x, q2)\n        >>> B3.orient_axis(B2, N.y, q3)\n        >>> B3.dcm(N).simplify()\n        Matrix([\n        [ sin(q1)*sin(q2)*sin(q3) + cos(q1)*cos(q3), sin(q1)*cos(q2), sin(q1)*sin(q2)*cos(q3) - sin(q3)*cos(q1)],\n        [-sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1), cos(q1)*cos(q2), sin(q1)*sin(q3) + sin(q2)*cos(q1)*cos(q3)],\n        [                           sin(q3)*cos(q2),        -sin(q2),                           cos(q2)*cos(q3)]])\n\n        It is worth noting that space-fixed and body-fixed rotations are\n        related by the order of the rotations, i.e. the reverse order of body\n        fixed will give space fixed and vice versa.\n\n        >>> B.orient_space_fixed(N, (q1, q2, q3), \'231\')\n        >>> B.dcm(N)\n        Matrix([\n        [cos(q1)*cos(q2), sin(q1)*sin(q3) + sin(q2)*cos(q1)*cos(q3), -sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1)],\n        [       -sin(q2),                           cos(q2)*cos(q3),                            sin(q3)*cos(q2)],\n        [sin(q1)*cos(q2), sin(q1)*sin(q2)*cos(q3) - sin(q3)*cos(q1),  sin(q1)*sin(q2)*sin(q3) + cos(q1)*cos(q3)]])\n\n        >>> B.orient_body_fixed(N, (q3, q2, q1), \'132\')\n        >>> B.dcm(N)\n        Matrix([\n        [cos(q1)*cos(q2), sin(q1)*sin(q3) + sin(q2)*cos(q1)*cos(q3), -sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1)],\n        [       -sin(q2),                           cos(q2)*cos(q3),                            sin(q3)*cos(q2)],\n        [sin(q1)*cos(q2), sin(q1)*sin(q2)*cos(q3) - sin(q3)*cos(q1),  sin(q1)*sin(q2)*sin(q3) + cos(q1)*cos(q3)]])\n\n        '
        from sympy.physics.vector.functions import dynamicsymbols
        _check_frame(parent)
        (amounts, rot_order, rot_matrices) = self._parse_consecutive_rotations(angles, rotation_order)
        self._dcm(parent, rot_matrices[2] * rot_matrices[1] * rot_matrices[0])
        rot_vecs = [zeros(3, 1) for _ in range(3)]
        for (i, order) in enumerate(rot_order):
            rot_vecs[i][order - 1] = amounts[i].diff(dynamicsymbols._t)
        (u1, u2, u3) = rot_vecs[0] + rot_matrices[0].T * (rot_vecs[1] + rot_matrices[1].T * rot_vecs[2])
        wvec = u1 * self.x + u2 * self.y + u3 * self.z
        self._ang_vel_dict.update({parent: wvec})
        parent._ang_vel_dict.update({self: -wvec})
        self._var_dict = {}

    def orient_quaternion(self, parent, numbers):
        if False:
            for i in range(10):
                print('nop')
        "Sets the orientation of this reference frame relative to a parent\n        reference frame via an orientation quaternion. An orientation\n        quaternion is defined as a finite rotation a unit vector, ``(lambda_x,\n        lambda_y, lambda_z)``, by an angle ``theta``. The orientation\n        quaternion is described by four parameters:\n\n        - ``q0 = cos(theta/2)``\n        - ``q1 = lambda_x*sin(theta/2)``\n        - ``q2 = lambda_y*sin(theta/2)``\n        - ``q3 = lambda_z*sin(theta/2)``\n\n        See `Quaternions and Spatial Rotation\n        <https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation>`_ on\n        Wikipedia for more information.\n\n        Parameters\n        ==========\n        parent : ReferenceFrame\n            Reference frame that this reference frame will be rotated relative\n            to.\n        numbers : 4-tuple of sympifiable\n            The four quaternion scalar numbers as defined above: ``q0``,\n            ``q1``, ``q2``, ``q3``.\n\n        Warns\n        ======\n\n        UserWarning\n            If the orientation creates a kinematic loop.\n\n        Examples\n        ========\n\n        Setup variables for the examples:\n\n        >>> from sympy import symbols\n        >>> from sympy.physics.vector import ReferenceFrame\n        >>> q0, q1, q2, q3 = symbols('q0 q1 q2 q3')\n        >>> N = ReferenceFrame('N')\n        >>> B = ReferenceFrame('B')\n\n        Set the orientation:\n\n        >>> B.orient_quaternion(N, (q0, q1, q2, q3))\n        >>> B.dcm(N)\n        Matrix([\n        [q0**2 + q1**2 - q2**2 - q3**2,             2*q0*q3 + 2*q1*q2,            -2*q0*q2 + 2*q1*q3],\n        [           -2*q0*q3 + 2*q1*q2, q0**2 - q1**2 + q2**2 - q3**2,             2*q0*q1 + 2*q2*q3],\n        [            2*q0*q2 + 2*q1*q3,            -2*q0*q1 + 2*q2*q3, q0**2 - q1**2 - q2**2 + q3**2]])\n\n        "
        from sympy.physics.vector.functions import dynamicsymbols
        _check_frame(parent)
        numbers = list(numbers)
        for (i, v) in enumerate(numbers):
            if not isinstance(v, Vector):
                numbers[i] = sympify(v)
        if not isinstance(numbers, (list, tuple)) & (len(numbers) == 4):
            raise TypeError('Amounts are a list or tuple of length 4')
        (q0, q1, q2, q3) = numbers
        parent_orient_quaternion = Matrix([[q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2, 2 * (q1 * q2 - q0 * q3), 2 * (q0 * q2 + q1 * q3)], [2 * (q1 * q2 + q0 * q3), q0 ** 2 - q1 ** 2 + q2 ** 2 - q3 ** 2, 2 * (q2 * q3 - q0 * q1)], [2 * (q1 * q3 - q0 * q2), 2 * (q0 * q1 + q2 * q3), q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2]])
        self._dcm(parent, parent_orient_quaternion)
        t = dynamicsymbols._t
        (q0, q1, q2, q3) = numbers
        q0d = diff(q0, t)
        q1d = diff(q1, t)
        q2d = diff(q2, t)
        q3d = diff(q3, t)
        w1 = 2 * (q1d * q0 + q2d * q3 - q3d * q2 - q0d * q1)
        w2 = 2 * (q2d * q0 + q3d * q1 - q1d * q3 - q0d * q2)
        w3 = 2 * (q3d * q0 + q1d * q2 - q2d * q1 - q0d * q3)
        wvec = Vector([(Matrix([w1, w2, w3]), self)])
        self._ang_vel_dict.update({parent: wvec})
        parent._ang_vel_dict.update({self: -wvec})
        self._var_dict = {}

    def orient(self, parent, rot_type, amounts, rot_order=''):
        if False:
            print('Hello World!')
        'Sets the orientation of this reference frame relative to another\n        (parent) reference frame.\n\n        .. note:: It is now recommended to use the ``.orient_axis,\n           .orient_body_fixed, .orient_space_fixed, .orient_quaternion``\n           methods for the different rotation types.\n\n        Parameters\n        ==========\n\n        parent : ReferenceFrame\n            Reference frame that this reference frame will be rotated relative\n            to.\n        rot_type : str\n            The method used to generate the direction cosine matrix. Supported\n            methods are:\n\n            - ``\'Axis\'``: simple rotations about a single common axis\n            - ``\'DCM\'``: for setting the direction cosine matrix directly\n            - ``\'Body\'``: three successive rotations about new intermediate\n              axes, also called "Euler and Tait-Bryan angles"\n            - ``\'Space\'``: three successive rotations about the parent\n              frames\' unit vectors\n            - ``\'Quaternion\'``: rotations defined by four parameters which\n              result in a singularity free direction cosine matrix\n\n        amounts :\n            Expressions defining the rotation angles or direction cosine\n            matrix. These must match the ``rot_type``. See examples below for\n            details. The input types are:\n\n            - ``\'Axis\'``: 2-tuple (expr/sym/func, Vector)\n            - ``\'DCM\'``: Matrix, shape(3,3)\n            - ``\'Body\'``: 3-tuple of expressions, symbols, or functions\n            - ``\'Space\'``: 3-tuple of expressions, symbols, or functions\n            - ``\'Quaternion\'``: 4-tuple of expressions, symbols, or\n              functions\n\n        rot_order : str or int, optional\n            If applicable, the order of the successive of rotations. The string\n            ``\'123\'`` and integer ``123`` are equivalent, for example. Required\n            for ``\'Body\'`` and ``\'Space\'``.\n\n        Warns\n        ======\n\n        UserWarning\n            If the orientation creates a kinematic loop.\n\n        '
        _check_frame(parent)
        approved_orders = ('123', '231', '312', '132', '213', '321', '121', '131', '212', '232', '313', '323', '')
        rot_order = translate(str(rot_order), 'XYZxyz', '123123')
        rot_type = rot_type.upper()
        if rot_order not in approved_orders:
            raise TypeError('The supplied order is not an approved type')
        if rot_type == 'AXIS':
            self.orient_axis(parent, amounts[1], amounts[0])
        elif rot_type == 'DCM':
            self.orient_explicit(parent, amounts)
        elif rot_type == 'BODY':
            self.orient_body_fixed(parent, amounts, rot_order)
        elif rot_type == 'SPACE':
            self.orient_space_fixed(parent, amounts, rot_order)
        elif rot_type == 'QUATERNION':
            self.orient_quaternion(parent, amounts)
        else:
            raise NotImplementedError('That is not an implemented rotation')

    def orientnew(self, newname, rot_type, amounts, rot_order='', variables=None, indices=None, latexs=None):
        if False:
            while True:
                i = 10
        'Returns a new reference frame oriented with respect to this\n        reference frame.\n\n        See ``ReferenceFrame.orient()`` for detailed examples of how to orient\n        reference frames.\n\n        Parameters\n        ==========\n\n        newname : str\n            Name for the new reference frame.\n        rot_type : str\n            The method used to generate the direction cosine matrix. Supported\n            methods are:\n\n            - ``\'Axis\'``: simple rotations about a single common axis\n            - ``\'DCM\'``: for setting the direction cosine matrix directly\n            - ``\'Body\'``: three successive rotations about new intermediate\n              axes, also called "Euler and Tait-Bryan angles"\n            - ``\'Space\'``: three successive rotations about the parent\n              frames\' unit vectors\n            - ``\'Quaternion\'``: rotations defined by four parameters which\n              result in a singularity free direction cosine matrix\n\n        amounts :\n            Expressions defining the rotation angles or direction cosine\n            matrix. These must match the ``rot_type``. See examples below for\n            details. The input types are:\n\n            - ``\'Axis\'``: 2-tuple (expr/sym/func, Vector)\n            - ``\'DCM\'``: Matrix, shape(3,3)\n            - ``\'Body\'``: 3-tuple of expressions, symbols, or functions\n            - ``\'Space\'``: 3-tuple of expressions, symbols, or functions\n            - ``\'Quaternion\'``: 4-tuple of expressions, symbols, or\n              functions\n\n        rot_order : str or int, optional\n            If applicable, the order of the successive of rotations. The string\n            ``\'123\'`` and integer ``123`` are equivalent, for example. Required\n            for ``\'Body\'`` and ``\'Space\'``.\n        indices : tuple of str\n            Enables the reference frame\'s basis unit vectors to be accessed by\n            Python\'s square bracket indexing notation using the provided three\n            indice strings and alters the printing of the unit vectors to\n            reflect this choice.\n        latexs : tuple of str\n            Alters the LaTeX printing of the reference frame\'s basis unit\n            vectors to the provided three valid LaTeX strings.\n\n        Examples\n        ========\n\n        >>> from sympy import symbols\n        >>> from sympy.physics.vector import ReferenceFrame, vlatex\n        >>> q0, q1, q2, q3 = symbols(\'q0 q1 q2 q3\')\n        >>> N = ReferenceFrame(\'N\')\n\n        Create a new reference frame A rotated relative to N through a simple\n        rotation.\n\n        >>> A = N.orientnew(\'A\', \'Axis\', (q0, N.x))\n\n        Create a new reference frame B rotated relative to N through body-fixed\n        rotations.\n\n        >>> B = N.orientnew(\'B\', \'Body\', (q1, q2, q3), \'123\')\n\n        Create a new reference frame C rotated relative to N through a simple\n        rotation with unique indices and LaTeX printing.\n\n        >>> C = N.orientnew(\'C\', \'Axis\', (q0, N.x), indices=(\'1\', \'2\', \'3\'),\n        ... latexs=(r\'\\hat{\\mathbf{c}}_1\',r\'\\hat{\\mathbf{c}}_2\',\n        ... r\'\\hat{\\mathbf{c}}_3\'))\n        >>> C[\'1\']\n        C[\'1\']\n        >>> print(vlatex(C[\'1\']))\n        \\hat{\\mathbf{c}}_1\n\n        '
        newframe = self.__class__(newname, variables=variables, indices=indices, latexs=latexs)
        approved_orders = ('123', '231', '312', '132', '213', '321', '121', '131', '212', '232', '313', '323', '')
        rot_order = translate(str(rot_order), 'XYZxyz', '123123')
        rot_type = rot_type.upper()
        if rot_order not in approved_orders:
            raise TypeError('The supplied order is not an approved type')
        if rot_type == 'AXIS':
            newframe.orient_axis(self, amounts[1], amounts[0])
        elif rot_type == 'DCM':
            newframe.orient_explicit(self, amounts)
        elif rot_type == 'BODY':
            newframe.orient_body_fixed(self, amounts, rot_order)
        elif rot_type == 'SPACE':
            newframe.orient_space_fixed(self, amounts, rot_order)
        elif rot_type == 'QUATERNION':
            newframe.orient_quaternion(self, amounts)
        else:
            raise NotImplementedError('That is not an implemented rotation')
        return newframe

    def set_ang_acc(self, otherframe, value):
        if False:
            for i in range(10):
                print('nop')
        "Define the angular acceleration Vector in a ReferenceFrame.\n\n        Defines the angular acceleration of this ReferenceFrame, in another.\n        Angular acceleration can be defined with respect to multiple different\n        ReferenceFrames. Care must be taken to not create loops which are\n        inconsistent.\n\n        Parameters\n        ==========\n\n        otherframe : ReferenceFrame\n            A ReferenceFrame to define the angular acceleration in\n        value : Vector\n            The Vector representing angular acceleration\n\n        Examples\n        ========\n\n        >>> from sympy.physics.vector import ReferenceFrame\n        >>> N = ReferenceFrame('N')\n        >>> A = ReferenceFrame('A')\n        >>> V = 10 * N.x\n        >>> A.set_ang_acc(N, V)\n        >>> A.ang_acc_in(N)\n        10*N.x\n\n        "
        if value == 0:
            value = Vector(0)
        value = _check_vector(value)
        _check_frame(otherframe)
        self._ang_acc_dict.update({otherframe: value})
        otherframe._ang_acc_dict.update({self: -value})

    def set_ang_vel(self, otherframe, value):
        if False:
            i = 10
            return i + 15
        "Define the angular velocity vector in a ReferenceFrame.\n\n        Defines the angular velocity of this ReferenceFrame, in another.\n        Angular velocity can be defined with respect to multiple different\n        ReferenceFrames. Care must be taken to not create loops which are\n        inconsistent.\n\n        Parameters\n        ==========\n\n        otherframe : ReferenceFrame\n            A ReferenceFrame to define the angular velocity in\n        value : Vector\n            The Vector representing angular velocity\n\n        Examples\n        ========\n\n        >>> from sympy.physics.vector import ReferenceFrame\n        >>> N = ReferenceFrame('N')\n        >>> A = ReferenceFrame('A')\n        >>> V = 10 * N.x\n        >>> A.set_ang_vel(N, V)\n        >>> A.ang_vel_in(N)\n        10*N.x\n\n        "
        if value == 0:
            value = Vector(0)
        value = _check_vector(value)
        _check_frame(otherframe)
        self._ang_vel_dict.update({otherframe: value})
        otherframe._ang_vel_dict.update({self: -value})

    @property
    def x(self):
        if False:
            print('Hello World!')
        'The basis Vector for the ReferenceFrame, in the x direction. '
        return self._x

    @property
    def y(self):
        if False:
            return 10
        'The basis Vector for the ReferenceFrame, in the y direction. '
        return self._y

    @property
    def z(self):
        if False:
            for i in range(10):
                print('nop')
        'The basis Vector for the ReferenceFrame, in the z direction. '
        return self._z

    @property
    def xx(self):
        if False:
            for i in range(10):
                print('nop')
        'Unit dyad of basis Vectors x and x for the ReferenceFrame.'
        return Vector.outer(self.x, self.x)

    @property
    def xy(self):
        if False:
            i = 10
            return i + 15
        'Unit dyad of basis Vectors x and y for the ReferenceFrame.'
        return Vector.outer(self.x, self.y)

    @property
    def xz(self):
        if False:
            i = 10
            return i + 15
        'Unit dyad of basis Vectors x and z for the ReferenceFrame.'
        return Vector.outer(self.x, self.z)

    @property
    def yx(self):
        if False:
            while True:
                i = 10
        'Unit dyad of basis Vectors y and x for the ReferenceFrame.'
        return Vector.outer(self.y, self.x)

    @property
    def yy(self):
        if False:
            for i in range(10):
                print('nop')
        'Unit dyad of basis Vectors y and y for the ReferenceFrame.'
        return Vector.outer(self.y, self.y)

    @property
    def yz(self):
        if False:
            return 10
        'Unit dyad of basis Vectors y and z for the ReferenceFrame.'
        return Vector.outer(self.y, self.z)

    @property
    def zx(self):
        if False:
            while True:
                i = 10
        'Unit dyad of basis Vectors z and x for the ReferenceFrame.'
        return Vector.outer(self.z, self.x)

    @property
    def zy(self):
        if False:
            return 10
        'Unit dyad of basis Vectors z and y for the ReferenceFrame.'
        return Vector.outer(self.z, self.y)

    @property
    def zz(self):
        if False:
            while True:
                i = 10
        'Unit dyad of basis Vectors z and z for the ReferenceFrame.'
        return Vector.outer(self.z, self.z)

    @property
    def u(self):
        if False:
            return 10
        'Unit dyadic for the ReferenceFrame.'
        return self.xx + self.yy + self.zz

    def partial_velocity(self, frame, *gen_speeds):
        if False:
            while True:
                i = 10
        "Returns the partial angular velocities of this frame in the given\n        frame with respect to one or more provided generalized speeds.\n\n        Parameters\n        ==========\n        frame : ReferenceFrame\n            The frame with which the angular velocity is defined in.\n        gen_speeds : functions of time\n            The generalized speeds.\n\n        Returns\n        =======\n        partial_velocities : tuple of Vector\n            The partial angular velocity vectors corresponding to the provided\n            generalized speeds.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.vector import ReferenceFrame, dynamicsymbols\n        >>> N = ReferenceFrame('N')\n        >>> A = ReferenceFrame('A')\n        >>> u1, u2 = dynamicsymbols('u1, u2')\n        >>> A.set_ang_vel(N, u1 * A.x + u2 * N.y)\n        >>> A.partial_velocity(N, u1)\n        A.x\n        >>> A.partial_velocity(N, u1, u2)\n        (A.x, N.y)\n\n        "
        partials = [self.ang_vel_in(frame).diff(speed, frame, var_in_dcm=False) for speed in gen_speeds]
        if len(partials) == 1:
            return partials[0]
        else:
            return tuple(partials)

def _check_frame(other):
    if False:
        return 10
    from .vector import VectorTypeError
    if not isinstance(other, ReferenceFrame):
        raise VectorTypeError(other, ReferenceFrame('A'))