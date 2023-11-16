import collections
from abc import ABC, abstractmethod
import numpy as np
from scipy._lib._util import MapWrapper

class VertexBase(ABC):
    """
    Base class for a vertex.
    """

    def __init__(self, x, nn=None, index=None):
        if False:
            i = 10
            return i + 15
        '\n        Initiation of a vertex object.\n\n        Parameters\n        ----------\n        x : tuple or vector\n            The geometric location (domain).\n        nn : list, optional\n            Nearest neighbour list.\n        index : int, optional\n            Index of vertex.\n        '
        self.x = x
        self.hash = hash(self.x)
        if nn is not None:
            self.nn = set(nn)
        else:
            self.nn = set()
        self.index = index

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.hash

    def __getattr__(self, item):
        if False:
            return 10
        if item not in ['x_a']:
            raise AttributeError(f"{type(self)} object has no attribute '{item}'")
        if item == 'x_a':
            self.x_a = np.array(self.x)
            return self.x_a

    @abstractmethod
    def connect(self, v):
        if False:
            while True:
                i = 10
        raise NotImplementedError('This method is only implemented with an associated child of the base class.')

    @abstractmethod
    def disconnect(self, v):
        if False:
            print('Hello World!')
        raise NotImplementedError('This method is only implemented with an associated child of the base class.')

    def star(self):
        if False:
            return 10
        'Returns the star domain ``st(v)`` of the vertex.\n\n        Parameters\n        ----------\n        v :\n            The vertex ``v`` in ``st(v)``\n\n        Returns\n        -------\n        st : set\n            A set containing all the vertices in ``st(v)``\n        '
        self.st = self.nn
        self.st.add(self)
        return self.st

class VertexScalarField(VertexBase):
    """
    Add homology properties of a scalar field f: R^n --> R associated with
    the geometry built from the VertexBase class
    """

    def __init__(self, x, field=None, nn=None, index=None, field_args=(), g_cons=None, g_cons_args=()):
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        x : tuple,\n            vector of vertex coordinates\n        field : callable, optional\n            a scalar field f: R^n --> R associated with the geometry\n        nn : list, optional\n            list of nearest neighbours\n        index : int, optional\n            index of the vertex\n        field_args : tuple, optional\n            additional arguments to be passed to field\n        g_cons : callable, optional\n            constraints on the vertex\n        g_cons_args : tuple, optional\n            additional arguments to be passed to g_cons\n\n        '
        super().__init__(x, nn=nn, index=index)
        self.check_min = True
        self.check_max = True

    def connect(self, v):
        if False:
            for i in range(10):
                print('nop')
        'Connects self to another vertex object v.\n\n        Parameters\n        ----------\n        v : VertexBase or VertexScalarField object\n        '
        if v is not self and v not in self.nn:
            self.nn.add(v)
            v.nn.add(self)
            self.check_min = True
            self.check_max = True
            v.check_min = True
            v.check_max = True

    def disconnect(self, v):
        if False:
            print('Hello World!')
        if v in self.nn:
            self.nn.remove(v)
            v.nn.remove(self)
            self.check_min = True
            self.check_max = True
            v.check_min = True
            v.check_max = True

    def minimiser(self):
        if False:
            while True:
                i = 10
        'Check whether this vertex is strictly less than all its\n           neighbours'
        if self.check_min:
            self._min = all((self.f < v.f for v in self.nn))
            self.check_min = False
        return self._min

    def maximiser(self):
        if False:
            while True:
                i = 10
        '\n        Check whether this vertex is strictly greater than all its\n        neighbours.\n        '
        if self.check_max:
            self._max = all((self.f > v.f for v in self.nn))
            self.check_max = False
        return self._max

class VertexVectorField(VertexBase):
    """
    Add homology properties of a scalar field f: R^n --> R^m associated with
    the geometry built from the VertexBase class.
    """

    def __init__(self, x, sfield=None, vfield=None, field_args=(), vfield_args=(), g_cons=None, g_cons_args=(), nn=None, index=None):
        if False:
            while True:
                i = 10
        super().__init__(x, nn=nn, index=index)
        raise NotImplementedError('This class is still a work in progress')

class VertexCacheBase:
    """Base class for a vertex cache for a simplicial complex."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.cache = collections.OrderedDict()
        self.nfev = 0
        self.index = -1

    def __iter__(self):
        if False:
            return 10
        for v in self.cache:
            yield self.cache[v]
        return

    def size(self):
        if False:
            i = 10
            return i + 15
        'Returns the size of the vertex cache.'
        return self.index + 1

    def print_out(self):
        if False:
            for i in range(10):
                print('nop')
        headlen = len(f'Vertex cache of size: {len(self.cache)}:')
        print('=' * headlen)
        print(f'Vertex cache of size: {len(self.cache)}:')
        print('=' * headlen)
        for v in self.cache:
            self.cache[v].print_out()

class VertexCube(VertexBase):
    """Vertex class to be used for a pure simplicial complex with no associated
    differential geometry (single level domain that exists in R^n)"""

    def __init__(self, x, nn=None, index=None):
        if False:
            print('Hello World!')
        super().__init__(x, nn=nn, index=index)

    def connect(self, v):
        if False:
            while True:
                i = 10
        if v is not self and v not in self.nn:
            self.nn.add(v)
            v.nn.add(self)

    def disconnect(self, v):
        if False:
            return 10
        if v in self.nn:
            self.nn.remove(v)
            v.nn.remove(self)

class VertexCacheIndex(VertexCacheBase):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Class for a vertex cache for a simplicial complex without an associated\n        field. Useful only for building and visualising a domain complex.\n\n        Parameters\n        ----------\n        '
        super().__init__()
        self.Vertex = VertexCube

    def __getitem__(self, x, nn=None):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self.cache[x]
        except KeyError:
            self.index += 1
            xval = self.Vertex(x, index=self.index)
            self.cache[x] = xval
            return self.cache[x]

class VertexCacheField(VertexCacheBase):

    def __init__(self, field=None, field_args=(), g_cons=None, g_cons_args=(), workers=1):
        if False:
            print('Hello World!')
        '\n        Class for a vertex cache for a simplicial complex with an associated\n        field.\n\n        Parameters\n        ----------\n        field : callable\n            Scalar or vector field callable.\n        field_args : tuple, optional\n            Any additional fixed parameters needed to completely specify the\n            field function\n        g_cons : dict or sequence of dict, optional\n            Constraints definition.\n            Function(s) ``R**n`` in the form::\n        g_cons_args : tuple, optional\n            Any additional fixed parameters needed to completely specify the\n            constraint functions\n        workers : int  optional\n            Uses `multiprocessing.Pool <multiprocessing>`) to compute the field\n             functions in parallel.\n\n        '
        super().__init__()
        self.index = -1
        self.Vertex = VertexScalarField
        self.field = field
        self.field_args = field_args
        self.wfield = FieldWrapper(field, field_args)
        self.g_cons = g_cons
        self.g_cons_args = g_cons_args
        self.wgcons = ConstraintWrapper(g_cons, g_cons_args)
        self.gpool = set()
        self.fpool = set()
        self.sfc_lock = False
        self.workers = workers
        self._mapwrapper = MapWrapper(workers)
        if workers == 1:
            self.process_gpool = self.proc_gpool
            if g_cons is None:
                self.process_fpool = self.proc_fpool_nog
            else:
                self.process_fpool = self.proc_fpool_g
        else:
            self.process_gpool = self.pproc_gpool
            if g_cons is None:
                self.process_fpool = self.pproc_fpool_nog
            else:
                self.process_fpool = self.pproc_fpool_g

    def __getitem__(self, x, nn=None):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self.cache[x]
        except KeyError:
            self.index += 1
            xval = self.Vertex(x, field=self.field, nn=nn, index=self.index, field_args=self.field_args, g_cons=self.g_cons, g_cons_args=self.g_cons_args)
            self.cache[x] = xval
            self.gpool.add(xval)
            self.fpool.add(xval)
            return self.cache[x]

    def __getstate__(self):
        if False:
            print('Hello World!')
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def process_pools(self):
        if False:
            print('Hello World!')
        if self.g_cons is not None:
            self.process_gpool()
        self.process_fpool()
        self.proc_minimisers()

    def feasibility_check(self, v):
        if False:
            while True:
                i = 10
        v.feasible = True
        for (g, args) in zip(self.g_cons, self.g_cons_args):
            if np.any(g(v.x_a, *args) < 0.0):
                v.f = np.inf
                v.feasible = False
                break

    def compute_sfield(self, v):
        if False:
            return 10
        'Compute the scalar field values of a vertex object `v`.\n\n        Parameters\n        ----------\n        v : VertexBase or VertexScalarField object\n        '
        try:
            v.f = self.field(v.x_a, *self.field_args)
            self.nfev += 1
        except AttributeError:
            v.f = np.inf
        if np.isnan(v.f):
            v.f = np.inf

    def proc_gpool(self):
        if False:
            print('Hello World!')
        'Process all constraints.'
        if self.g_cons is not None:
            for v in self.gpool:
                self.feasibility_check(v)
        self.gpool = set()

    def pproc_gpool(self):
        if False:
            i = 10
            return i + 15
        'Process all constraints in parallel.'
        gpool_l = []
        for v in self.gpool:
            gpool_l.append(v.x_a)
        G = self._mapwrapper(self.wgcons.gcons, gpool_l)
        for (v, g) in zip(self.gpool, G):
            v.feasible = g

    def proc_fpool_g(self):
        if False:
            i = 10
            return i + 15
        'Process all field functions with constraints supplied.'
        for v in self.fpool:
            if v.feasible:
                self.compute_sfield(v)
        self.fpool = set()

    def proc_fpool_nog(self):
        if False:
            while True:
                i = 10
        'Process all field functions with no constraints supplied.'
        for v in self.fpool:
            self.compute_sfield(v)
        self.fpool = set()

    def pproc_fpool_g(self):
        if False:
            i = 10
            return i + 15
        '\n        Process all field functions with constraints supplied in parallel.\n        '
        self.wfield.func
        fpool_l = []
        for v in self.fpool:
            if v.feasible:
                fpool_l.append(v.x_a)
            else:
                v.f = np.inf
        F = self._mapwrapper(self.wfield.func, fpool_l)
        for (va, f) in zip(fpool_l, F):
            vt = tuple(va)
            self[vt].f = f
            self.nfev += 1
        self.fpool = set()

    def pproc_fpool_nog(self):
        if False:
            while True:
                i = 10
        '\n        Process all field functions with no constraints supplied in parallel.\n        '
        self.wfield.func
        fpool_l = []
        for v in self.fpool:
            fpool_l.append(v.x_a)
        F = self._mapwrapper(self.wfield.func, fpool_l)
        for (va, f) in zip(fpool_l, F):
            vt = tuple(va)
            self[vt].f = f
            self.nfev += 1
        self.fpool = set()

    def proc_minimisers(self):
        if False:
            i = 10
            return i + 15
        'Check for minimisers.'
        for v in self:
            v.minimiser()
            v.maximiser()

class ConstraintWrapper:
    """Object to wrap constraints to pass to `multiprocessing.Pool`."""

    def __init__(self, g_cons, g_cons_args):
        if False:
            for i in range(10):
                print('nop')
        self.g_cons = g_cons
        self.g_cons_args = g_cons_args

    def gcons(self, v_x_a):
        if False:
            while True:
                i = 10
        vfeasible = True
        for (g, args) in zip(self.g_cons, self.g_cons_args):
            if np.any(g(v_x_a, *args) < 0.0):
                vfeasible = False
                break
        return vfeasible

class FieldWrapper:
    """Object to wrap field to pass to `multiprocessing.Pool`."""

    def __init__(self, field, field_args):
        if False:
            while True:
                i = 10
        self.field = field
        self.field_args = field_args

    def func(self, v_x_a):
        if False:
            print('Hello World!')
        try:
            v_f = self.field(v_x_a, *self.field_args)
        except Exception:
            v_f = np.inf
        if np.isnan(v_f):
            v_f = np.inf
        return v_f