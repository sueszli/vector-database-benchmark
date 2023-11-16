"""
Raw Point Collection

This collection provides very fast points. Output quality is ugly so it must be
used at small size only (2/3 pixels). You've been warned.
"""
from __future__ import division
import numpy as np
from ... import glsl
from .collection import Collection
from ..transforms import NullTransform

class RawPointCollection(Collection):
    """
    Raw Point Collection

    This collection provides very fast points. Output quality is ugly so it
    must be used at small size only (2/3 pixels). You've been warned.
    """

    def __init__(self, user_dtype=None, transform=None, vertex=None, fragment=None, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Initialize the collection.\n\n        Parameters\n        ----------\n        user_dtype: list\n            The base dtype can be completed (appended) by the used_dtype. It\n            only make sense if user also provide vertex and/or fragment shaders\n\n        transform : Transform instance\n            Used to define the transform(vec4) function\n\n        vertex: string\n            Vertex shader code\n\n        fragment: string\n            Fragment  shader code\n\n        color : string\n            'local', 'shared' or 'global'\n        "
        base_dtype = [('position', (np.float32, 3), '!local', (0, 0, 0)), ('size', (np.float32, 1), 'global', 3.0), ('color', (np.float32, 4), 'global', (0, 0, 0, 1))]
        dtype = base_dtype
        if user_dtype:
            dtype.extend(user_dtype)
        if vertex is None:
            vertex = glsl.get('collections/raw-point.vert')
        if transform is None:
            transform = NullTransform()
        self.transform = transform
        if fragment is None:
            fragment = glsl.get('collections/raw-point.frag')
        Collection.__init__(self, dtype=dtype, itype=None, mode='points', vertex=vertex, fragment=fragment, **kwargs)
        program = self._programs[0]
        program.vert['transform'] = self.transform

    def append(self, P, itemsize=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Append a new set of vertices to the collection.\n\n        For kwargs argument, n is the number of vertices (local) or the number\n        of item (shared)\n\n        Parameters\n        ----------\n        P : np.array\n            Vertices positions of the points(s) to be added\n\n        itemsize: int or None\n            Size of an individual path\n\n        color : list, array or 4-tuple\n           Path color\n        '
        itemsize = itemsize or 1
        itemcount = len(P) // itemsize
        V = np.empty(len(P), dtype=self.vtype)
        for name in self.vtype.names:
            if name not in ['position', 'collection_index']:
                V[name] = kwargs.get(name, self._defaults[name])
        V['position'] = P
        if self.utype:
            U = np.zeros(itemcount, dtype=self.utype)
            for name in self.utype.names:
                if name not in ['__unused__']:
                    U[name] = kwargs.get(name, self._defaults[name])
        else:
            U = None
        Collection.append(self, vertices=V, uniforms=U, itemsize=itemsize)