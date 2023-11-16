"""
Antigrain Geometry Path Collection

This collection provides antialiased and accurate paths with caps and joins. It
is memory hungry (x8) and slow (x.25) so it is to be used sparingly, mainly for
thick paths where quality is critical.
"""
import numpy as np
from ... import glsl
from ... import gloo
from .collection import Collection
from ..transforms import NullTransform

class AggPathCollection(Collection):
    """
    Antigrain Geometry Path Collection

    This collection provides antialiased and accurate paths with caps and
    joins. It is memory hungry (x8) and slow (x.25) so it is to be used
    sparingly, mainly for thick paths where quality is critical.
    """

    def __init__(self, user_dtype=None, transform=None, vertex=None, fragment=None, **kwargs):
        if False:
            return 10
        "\n        Initialize the collection.\n\n        Parameters\n        ----------\n        user_dtype: list\n            The base dtype can be completed (appended) by the used_dtype. It\n            only make sense if user also provide vertex and/or fragment shaders\n\n        transform : Transform instance\n            Used to define the transform(vec4) function\n\n        vertex: string\n            Vertex shader code\n\n        fragment: string\n            Fragment  shader code\n\n        caps : string\n            'local', 'shared' or 'global'\n\n        join : string\n            'local', 'shared' or 'global'\n\n        color : string\n            'local', 'shared' or 'global'\n\n        miter_limit : string\n            'local', 'shared' or 'global'\n\n        linewidth : string\n            'local', 'shared' or 'global'\n\n        antialias : string\n            'local', 'shared' or 'global'\n        "
        base_dtype = [('p0', (np.float32, 3), '!local', (0, 0, 0)), ('p1', (np.float32, 3), '!local', (0, 0, 0)), ('p2', (np.float32, 3), '!local', (0, 0, 0)), ('p3', (np.float32, 3), '!local', (0, 0, 0)), ('uv', (np.float32, 2), '!local', (0, 0)), ('caps', (np.float32, 2), 'global', (0, 0)), ('join', (np.float32, 1), 'global', 0), ('color', (np.float32, 4), 'global', (0, 0, 0, 1)), ('miter_limit', (np.float32, 1), 'global', 4), ('linewidth', (np.float32, 1), 'global', 1), ('antialias', (np.float32, 1), 'global', 1), ('viewport', (np.float32, 4), 'global', (0, 0, 512, 512))]
        dtype = base_dtype
        if user_dtype:
            dtype.extend(user_dtype)
        if vertex is None:
            vertex = glsl.get('collections/agg-path.vert')
        if transform is None:
            transform = NullTransform()
        self.transform = transform
        if fragment is None:
            fragment = glsl.get('collections/agg-path.frag')
        Collection.__init__(self, dtype=dtype, itype=np.uint32, mode='triangles', vertex=vertex, fragment=fragment, **kwargs)
        self._programs[0].vert['transform'] = self.transform

    def append(self, P, closed=False, itemsize=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Append a new set of vertices to the collection.\n\n        For kwargs argument, n is the number of vertices (local) or the number\n        of item (shared)\n\n        Parameters\n        ----------\n        P : np.array\n            Vertices positions of the path(s) to be added\n\n        closed: bool\n            Whether path(s) is/are closed\n\n        itemsize: int or None\n            Size of an individual path\n\n        caps : list, array or 2-tuple\n           Path start /end cap\n\n        join : list, array or float\n           path segment join\n\n        color : list, array or 4-tuple\n           Path color\n\n        miter_limit : list, array or float\n           Miter limit for join\n\n        linewidth : list, array or float\n           Path linewidth\n\n        antialias : list, array or float\n           Path antialias area\n        '
        itemsize = int(itemsize or len(P))
        itemcount = len(P) // itemsize
        (n, p) = (len(P), P.shape[-1])
        Z = np.tile(P, 2).reshape(2 * len(P), p)
        V = np.empty(n, dtype=self.vtype)
        V['p0'][1:-1] = Z[0::2][:-2]
        V['p1'][:-1] = Z[1::2][:-1]
        V['p2'][:-1] = Z[1::2][+1:]
        V['p3'][:-2] = Z[0::2][+2:]
        for name in self.vtype.names:
            if name not in ['collection_index', 'p0', 'p1', 'p2', 'p3']:
                V[name] = kwargs.get(name, self._defaults[name])
        V = V.reshape(n // itemsize, itemsize)[:, :-1]
        if closed:
            V['p0'][:, 0] = V['p2'][:, -1]
            V['p3'][:, -1] = V['p1'][:, 0]
        else:
            V['p0'][:, 0] = V['p1'][:, 0]
            V['p3'][:, -1] = V['p2'][:, -1]
        V = V.ravel()
        V = np.repeat(V, 4, axis=0).reshape((len(V), 4))
        V['uv'] = ((-1, -1), (-1, +1), (+1, -1), (+1, +1))
        V = V.ravel()
        n = itemsize
        if closed:
            idxs = np.resize(np.array([0, 1, 2, 1, 2, 3], dtype=np.uint32), n * 2 * 3)
            idxs += np.repeat(4 * np.arange(n, dtype=np.uint32), 6)
            idxs[-6:] = (4 * n - 6, 4 * n - 5, 0, 4 * n - 5, 0, 1)
        else:
            idxs = np.resize(np.array([0, 1, 2, 1, 2, 3], dtype=np.uint32), (n - 1) * 2 * 3)
            idxs += np.repeat(4 * np.arange(n - 1, dtype=np.uint32), 6)
        idxs = idxs.ravel()
        if self.utype:
            U = np.zeros(itemcount, dtype=self.utype)
            for name in self.utype.names:
                if name not in ['__unused__']:
                    U[name] = kwargs.get(name, self._defaults[name])
        else:
            U = None
        Collection.append(self, vertices=V, uniforms=U, indices=idxs, itemsize=itemsize * 4 - 4)

    def draw(self, mode='triangles'):
        if False:
            i = 10
            return i + 15
        'Draw collection'
        gloo.set_depth_mask(0)
        Collection.draw(self, mode)
        gloo.set_depth_mask(1)