import numpy as np
from ... import glsl
from .collection import Collection
from ..transforms import NullTransform

class RawPathCollection(Collection):
    """
    """

    def __init__(self, user_dtype=None, transform=None, vertex=None, fragment=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Initialize the collection.\n\n        Parameters\n        ----------\n        user_dtype: list\n            The base dtype can be completed (appended) by the used_dtype. It\n            only make sense if user also provide vertex and/or fragment shaders\n\n        transform : Transform instance\n            Used to define the transform(vec4) function\n\n        vertex: string\n            Vertex shader code\n\n        fragment: string\n            Fragment  shader code\n\n        color : string\n            'local', 'shared' or 'global'\n        "
        base_dtype = [('position', (np.float32, 3), '!local', (0, 0, 0)), ('id', (np.float32, 1), '!local', 0), ('color', (np.float32, 4), 'local', (0, 0, 0, 1)), ('linewidth', (np.float32, 1), 'global', 1), ('viewport', (np.float32, 4), 'global', (0, 0, 512, 512))]
        dtype = base_dtype
        if user_dtype:
            dtype.extend(user_dtype)
        if vertex is None:
            vertex = glsl.get('collections/raw-path.vert')
        if transform is None:
            transform = NullTransform()
        self.transform = transform
        if fragment is None:
            fragment = glsl.get('collections/raw-path.frag')
        vertex = transform + vertex
        Collection.__init__(self, dtype=dtype, itype=None, mode='line_strip', vertex=vertex, fragment=fragment, **kwargs)
        self._programs[0].vert['transform'] = self.transform

    def append(self, P, closed=False, itemsize=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Append a new set of vertices to the collection.\n\n        For kwargs argument, n is the number of vertices (local) or the number\n        of item (shared)\n\n        Parameters\n        ----------\n        P : np.array\n            Vertices positions of the path(s) to be added\n\n        closed: bool\n            Whether path(s) is/are closed\n\n        itemsize: int or None\n            Size of an individual path\n\n        color : list, array or 4-tuple\n           Path color\n        '
        itemsize = itemsize or len(P)
        itemcount = len(P) / itemsize
        P = P.reshape(itemcount, itemsize, 3)
        if closed:
            V = np.empty((itemcount, itemsize + 3), dtype=self.vtype)
            for name in self.vtype.names:
                if name not in ['collection_index', 'position']:
                    V[name][1:-2] = kwargs.get(name, self._defaults[name])
            V['position'][:, 1:-2] = P
            V['position'][:, -2] = V['position'][:, 1]
        else:
            V = np.empty((itemcount, itemsize + 2), dtype=self.vtype)
            for name in self.vtype.names:
                if name not in ['collection_index', 'position']:
                    V[name][1:-1] = kwargs.get(name, self._defaults[name])
            V['position'][:, 1:-1] = P
        V['id'] = 1
        V[:, 0] = V[:, 1]
        V[:, -1] = V[:, -2]
        V['id'][:, 0] = 0
        V['id'][:, -1] = 0
        if self.utype:
            U = np.zeros(itemcount, dtype=self.utype)
            for name in self.utype.names:
                if name not in ['__unused__']:
                    U[name] = kwargs.get(name, self._defaults[name])
        else:
            U = None
        Collection.append(self, vertices=V, uniforms=U, itemsize=itemsize + 2 + closed)