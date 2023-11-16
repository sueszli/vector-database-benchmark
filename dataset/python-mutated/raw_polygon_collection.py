import numpy as np
from ... import glsl
from .collection import Collection
from ..transforms import NullTransform
from ...geometry import triangulate

class RawPolygonCollection(Collection):

    def __init__(self, user_dtype=None, transform=None, vertex=None, fragment=None, **kwargs):
        if False:
            while True:
                i = 10
        base_dtype = [('position', (np.float32, 3), '!local', (0, 0, 0)), ('color', (np.float32, 4), 'local', (0, 0, 0, 1))]
        dtype = base_dtype
        if user_dtype:
            dtype.extend(user_dtype)
        if vertex is None:
            vertex = glsl.get('collections/raw-triangle.vert')
        if transform is None:
            transform = NullTransform()
        self.transform = transform
        if fragment is None:
            fragment = glsl.get('collections/raw-triangle.frag')
        Collection.__init__(self, dtype=dtype, itype=np.uint32, mode='triangles', vertex=vertex, fragment=fragment, **kwargs)
        program = self._programs[0]
        program.vert['transform'] = self.transform

    def append(self, points, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Append a new set of vertices to the collection.\n\n        For kwargs argument, n is the number of vertices (local) or the number\n        of item (shared)\n\n        Parameters\n        ----------\n        points : np.array\n            Vertices composing the triangles\n\n        color : list, array or 4-tuple\n           Path color\n        '
        (vertices, indices) = triangulate(points)
        itemsize = len(vertices)
        itemcount = 1
        V = np.empty(itemcount * itemsize, dtype=self.vtype)
        for name in self.vtype.names:
            if name not in ['collection_index', 'position']:
                V[name] = kwargs.get(name, self._defaults[name])
        V['position'] = vertices
        if self.utype:
            U = np.zeros(itemcount, dtype=self.utype)
            for name in self.utype.names:
                if name not in ['__unused__']:
                    U[name] = kwargs.get(name, self._defaults[name])
        else:
            U = None
        Collection.append(self, vertices=V, uniforms=U, indices=np.array(indices).ravel(), itemsize=itemsize)