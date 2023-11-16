import numpy as np

def _fix_colors(colors):
    if False:
        print('Hello World!')
    colors = np.asarray(colors)
    if colors.ndim not in (2, 3):
        raise ValueError('colors must have 2 or 3 dimensions')
    if colors.shape[-1] not in (3, 4):
        raise ValueError('colors must have 3 or 4 elements')
    if colors.shape[-1] == 3:
        pad = np.ones((len(colors), 1), colors.dtype)
        if colors.ndim == 3:
            pad = pad[:, :, np.newaxis]
        colors = np.concatenate((colors, pad), axis=-1)
    return colors

def _compute_face_normals(vertices):
    if False:
        return 10
    if vertices.shape[1:] != (3, 3):
        raise ValueError(f'Expected (N, 3, 3) array of vertices repeated on the triangle corners, got {vertices.shape}.')
    edges1 = vertices[:, 1] - vertices[:, 0]
    edges2 = vertices[:, 2] - vertices[:, 0]
    return np.cross(edges1, edges2)

def _repeat_face_normals_on_corners(normals):
    if False:
        return 10
    if normals.shape[1:] != (3,):
        raise ValueError(f'Expected (F, 3) array of face normals, got {normals.shape}.')
    n_corners_in_face = 3
    new_shape = (normals.shape[0], n_corners_in_face, normals.shape[1])
    return np.repeat(normals, n_corners_in_face, axis=0).reshape(new_shape)

def _compute_vertex_normals(face_normals, faces, vertices):
    if False:
        while True:
            i = 10
    if face_normals.shape[1:] != (3,):
        raise ValueError(f'Expected (F, 3) array of face normals, got {face_normals.shape}.')
    if faces.shape[1:] != (3,):
        raise ValueError(f'Expected (F, 3) array of face vertex indices, got {faces.shape}.')
    if vertices.shape[1:] != (3,):
        raise ValueError(f'Expected (N, 3) array of vertices, got {vertices.shape}.')
    vertex_normals = np.zeros_like(vertices)
    n_corners_in_triangle = 3
    face_normals_repeated_on_corners = np.repeat(face_normals, n_corners_in_triangle, axis=0)
    np.add.at(vertex_normals, faces.ravel(), face_normals_repeated_on_corners)
    norms = np.sqrt((vertex_normals ** 2).sum(axis=1))
    nonzero_norms = norms > 0
    vertex_normals[nonzero_norms] /= norms[nonzero_norms][:, None]
    return vertex_normals

class MeshData(object):
    """
    Class for storing and operating on 3D mesh data.

    Parameters
    ----------
    vertices : ndarray, shape (Nv, 3)
        Vertex coordinates. If faces is not specified, then this will
        instead be interpreted as (Nf, 3, 3) array of coordinates.
    faces : ndarray, shape (Nf, 3)
        Indices into the vertex array.
    edges : None
        [not available yet]
    vertex_colors : ndarray, shape (Nv, 4)
        Vertex colors. If faces is not specified, this will be
        interpreted as (Nf, 3, 4) array of colors.
    face_colors : ndarray, shape (Nf, 4)
        Face colors.
    vertex_values : ndarray, shape (Nv,)
        Vertex values.

    Notes
    -----
    All arguments are optional.

    The object may contain:

    - list of vertex locations
    - list of edges
    - list of triangles
    - colors per vertex, edge, or tri
    - normals per vertex or tri

    This class handles conversion between the standard
    [list of vertices, list of faces] format (suitable for use with
    glDrawElements) and 'indexed' [list of vertices] format (suitable
    for use with glDrawArrays). It will automatically compute face normal
    vectors as well as averaged vertex normal vectors.

    The class attempts to be as efficient as possible in caching conversion
    results and avoiding unnecessary conversions.
    """

    def __init__(self, vertices=None, faces=None, edges=None, vertex_colors=None, face_colors=None, vertex_values=None):
        if False:
            return 10
        self._vertices = None
        self._vertices_indexed_by_faces = None
        self._vertices_indexed_by_edges = None
        self._faces = None
        self._edges = None
        self._edges_indexed_by_faces = None
        self._vertex_faces = None
        self._vertex_edges = None
        self._vertex_normals = None
        self._vertex_normals_indexed_by_faces = None
        self._vertex_colors = None
        self._vertex_colors_indexed_by_faces = None
        self._vertex_colors_indexed_by_edges = None
        self._vertex_values = None
        self._vertex_values_indexed_by_faces = None
        self._vertex_values_indexed_by_edges = None
        self._face_normals = None
        self._face_normals_indexed_by_faces = None
        self._face_colors = None
        self._face_colors_indexed_by_faces = None
        self._face_colors_indexed_by_edges = None
        self._edge_colors = None
        self._edge_colors_indexed_by_edges = None
        if vertices is not None:
            indexed = 'faces' if faces is None else None
            self.set_vertices(vertices, indexed=indexed)
            if faces is not None:
                self.set_faces(faces)
            if vertex_colors is not None:
                self.set_vertex_colors(vertex_colors, indexed=indexed)
            if face_colors is not None:
                self.set_face_colors(face_colors, indexed=indexed)
            if vertex_values is not None:
                self.set_vertex_values(vertex_values, indexed=indexed)

    def get_faces(self):
        if False:
            while True:
                i = 10
        'Array (Nf, 3) of vertex indices, three per triangular face.\n\n        If faces have not been computed for this mesh, returns None.\n        '
        return self._faces

    def get_edges(self, indexed=None):
        if False:
            print('Hello World!')
        "Edges of the mesh\n\n        Parameters\n        ----------\n        indexed : str | None\n           If indexed is None, return (Nf, 3) array of vertex indices,\n           two per edge in the mesh.\n           If indexed is 'faces', then return (Nf, 3, 2) array of vertex\n           indices with 3 edges per face, and two vertices per edge.\n\n        Returns\n        -------\n        edges : ndarray\n            The edges.\n        "
        if indexed is None:
            if self._edges is None:
                self._compute_edges(indexed=None)
            return self._edges
        elif indexed == 'faces':
            if self._edges_indexed_by_faces is None:
                self._compute_edges(indexed='faces')
            return self._edges_indexed_by_faces
        else:
            raise ValueError("Invalid indexing mode. Accepts: None, 'faces'")

    def set_faces(self, faces):
        if False:
            return 10
        'Set the faces\n\n        Parameters\n        ----------\n        faces : ndarray\n            (Nf, 3) array of faces. Each row in the array contains\n            three indices into the vertex array, specifying the three corners\n            of a triangular face.\n        '
        self._faces = faces
        self._edges = None
        self._edges_indexed_by_faces = None
        self._vertex_faces = None
        self._vertices_indexed_by_faces = None
        self.reset_normals()
        self._vertex_colors_indexed_by_faces = None
        self._face_colors_indexed_by_faces = None

    def get_vertices(self, indexed=None):
        if False:
            for i in range(10):
                print('nop')
        "Get the vertices\n\n        Parameters\n        ----------\n        indexed : str | None\n            If Note, return an array (N,3) of the positions of vertices in\n            the mesh. By default, each unique vertex appears only once.\n            If indexed is 'faces', then the array will instead contain three\n            vertices per face in the mesh (and a single vertex may appear more\n            than once in the array).\n\n        Returns\n        -------\n        vertices : ndarray\n            The vertices.\n        "
        if indexed is None:
            if self._vertices is None and self._vertices_indexed_by_faces is not None:
                self._compute_unindexed_vertices()
            return self._vertices
        elif indexed == 'faces':
            if self._vertices_indexed_by_faces is None and self._vertices is not None:
                self._vertices_indexed_by_faces = self._vertices[self.get_faces()]
            return self._vertices_indexed_by_faces
        else:
            raise ValueError("Invalid indexing mode. Accepts: None, 'faces'")

    def get_bounds(self):
        if False:
            return 10
        'Get the mesh bounds\n\n        Returns\n        -------\n        bounds : list\n            A list of tuples of mesh bounds.\n        '
        if self._vertices_indexed_by_faces is not None:
            v = self._vertices_indexed_by_faces
        elif self._vertices is not None:
            v = self._vertices
        else:
            return None
        bounds = [(v[:, ax].min(), v[:, ax].max()) for ax in range(v.shape[1])]
        return bounds

    def set_vertices(self, verts=None, indexed=None, reset_normals=True):
        if False:
            print('Hello World!')
        "Set the mesh vertices\n\n        Parameters\n        ----------\n        verts : ndarray | None\n            The array (Nv, 3) of vertex coordinates.\n        indexed : str | None\n            If indexed=='faces', then the data must have shape (Nf, 3, 3) and\n            is assumed to be already indexed as a list of faces. This will\n            cause any pre-existing normal vectors to be cleared unless\n            reset_normals=False.\n        reset_normals : bool\n            If True, reset the normals.\n        "
        if indexed is None:
            if verts is not None:
                self._vertices = verts
            self._vertices_indexed_by_faces = None
        elif indexed == 'faces':
            self._vertices = None
            if verts is not None:
                self._vertices_indexed_by_faces = verts
        else:
            raise ValueError("Invalid indexing mode. Accepts: None, 'faces'")
        if reset_normals:
            self.reset_normals()

    def reset_normals(self):
        if False:
            print('Hello World!')
        self._vertex_normals = None
        self._vertex_normals_indexed_by_faces = None
        self._face_normals = None
        self._face_normals_indexed_by_faces = None

    def has_face_indexed_data(self):
        if False:
            print('Hello World!')
        'Return True if this object already has vertex positions indexed\n        by face\n        '
        return self._vertices_indexed_by_faces is not None

    def has_edge_indexed_data(self):
        if False:
            i = 10
            return i + 15
        return self._vertices_indexed_by_edges is not None

    def has_vertex_color(self):
        if False:
            print('Hello World!')
        'Return True if this data set has vertex color information'
        for v in (self._vertex_colors, self._vertex_colors_indexed_by_faces, self._vertex_colors_indexed_by_edges):
            if v is not None:
                return True
        return False

    def has_vertex_value(self):
        if False:
            return 10
        'Return True if this data set has vertex value information'
        for v in (self._vertex_values, self._vertex_values_indexed_by_faces, self._vertex_values_indexed_by_edges):
            if v is not None:
                return True
        return False

    def has_face_color(self):
        if False:
            print('Hello World!')
        'Return True if this data set has face color information'
        for v in (self._face_colors, self._face_colors_indexed_by_faces, self._face_colors_indexed_by_edges):
            if v is not None:
                return True
        return False

    def get_face_normals(self, indexed=None):
        if False:
            print('Hello World!')
        "Get face normals\n\n        Parameters\n        ----------\n        indexed : str | None\n            If None, return an array (Nf, 3) of normal vectors for each face.\n            If 'faces', then instead return an indexed array (Nf, 3, 3)\n            (this is just the same array with each vector copied three times).\n\n        Returns\n        -------\n        normals : ndarray\n            The normals.\n        "
        if indexed not in (None, 'faces'):
            raise ValueError("Invalid indexing mode. Accepts: None, 'faces'")
        if self._face_normals is None:
            vertices = self.get_vertices(indexed='faces')
            self._face_normals = _compute_face_normals(vertices)
        if indexed == 'faces' and self._face_normals_indexed_by_faces is None:
            self._face_normals_indexed_by_faces = _repeat_face_normals_on_corners(self._face_normals)
        return self._face_normals if indexed is None else self._face_normals_indexed_by_faces

    def get_vertex_normals(self, indexed=None):
        if False:
            while True:
                i = 10
        "Get vertex normals\n\n        Parameters\n        ----------\n        indexed : str | None\n            If None, return an (N, 3) array of normal vectors with one entry\n            per unique vertex in the mesh. If indexed is 'faces', then the\n            array will contain three normal vectors per face (and some\n            vertices may be repeated).\n\n        Returns\n        -------\n        normals : ndarray\n            The normals.\n        "
        if indexed not in (None, 'faces'):
            raise ValueError("Invalid indexing mode. Accepts: None, 'faces'")
        if self._vertex_normals is None:
            face_normals = self.get_face_normals()
            faces = self.get_faces()
            vertices = self.get_vertices()
            self._vertex_normals = _compute_vertex_normals(face_normals, faces, vertices)
        if indexed is None:
            return self._vertex_normals
        elif indexed == 'faces':
            return self._vertex_normals[self.get_faces()]

    def get_vertex_colors(self, indexed=None):
        if False:
            i = 10
            return i + 15
        "Get vertex colors\n\n        Parameters\n        ----------\n        indexed : str | None\n            If None, return an array (Nv, 4) of vertex colors.\n            If indexed=='faces', then instead return an indexed array\n            (Nf, 3, 4).\n\n        Returns\n        -------\n        colors : ndarray\n            The vertex colors.\n        "
        if indexed is None:
            return self._vertex_colors
        elif indexed == 'faces':
            if self._vertex_colors_indexed_by_faces is None:
                self._vertex_colors_indexed_by_faces = self._vertex_colors[self.get_faces()]
            return self._vertex_colors_indexed_by_faces
        else:
            raise ValueError("Invalid indexing mode. Accepts: None, 'faces'")

    def get_vertex_values(self, indexed=None):
        if False:
            return 10
        "Get vertex colors\n\n        Parameters\n        ----------\n        indexed : str | None\n            If None, return an array (Nv,) of vertex values.\n            If indexed=='faces', then instead return an indexed array\n            (Nf, 3).\n\n        Returns\n        -------\n        values : ndarray\n            The vertex values.\n        "
        if indexed is None:
            return self._vertex_values
        elif indexed == 'faces':
            if self._vertex_values_indexed_by_faces is None:
                self._vertex_values_indexed_by_faces = self._vertex_values[self.get_faces()]
            return self._vertex_values_indexed_by_faces
        else:
            raise ValueError("Invalid indexing mode. Accepts: None, 'faces'")

    def set_vertex_colors(self, colors, indexed=None):
        if False:
            return 10
        "Set the vertex color array\n\n        Parameters\n        ----------\n        colors : array\n            Array of colors. Must have shape (Nv, 4) (indexing by vertex)\n            or shape (Nf, 3, 4) (vertices indexed by face).\n        indexed : str | None\n            Should be 'faces' if colors are indexed by faces.\n        "
        colors = _fix_colors(colors)
        if indexed is None:
            if colors.ndim != 2:
                raise ValueError('colors must be 2D if indexed is None')
            if colors.shape[0] != self.n_vertices:
                raise ValueError('incorrect number of colors %s, expected %s' % (colors.shape[0], self.n_vertices))
            self._vertex_colors = colors
            self._vertex_colors_indexed_by_faces = None
        elif indexed == 'faces':
            if colors.ndim != 3:
                raise ValueError('colors must be 3D if indexed is "faces"')
            if colors.shape[0] != self.n_faces:
                raise ValueError('incorrect number of faces')
            self._vertex_colors = None
            self._vertex_colors_indexed_by_faces = colors
        else:
            raise ValueError('indexed must be None or "faces"')

    def set_vertex_values(self, values, indexed=None):
        if False:
            while True:
                i = 10
        "Set the vertex value array\n\n        Parameters\n        ----------\n        values : array\n            Array of values. Must have shape (Nv,) (indexing by vertex)\n            or shape (Nf, 3) (vertices indexed by face).\n        indexed : str | None\n            Should be 'faces' if colors are indexed by faces.\n        "
        values = np.asarray(values)
        if indexed is None:
            if values.ndim != 1:
                raise ValueError('values must be 1D if indexed is None')
            if values.shape[0] != self.n_vertices:
                raise ValueError('incorrect number of colors %s, expected %s' % (values.shape[0], self.n_vertices))
            self._vertex_values = values
            self._vertex_values_indexed_by_faces = None
        elif indexed == 'faces':
            if values.ndim != 2:
                raise ValueError('values must be 3D if indexed is "faces"')
            if values.shape[0] != self.n_faces:
                raise ValueError('incorrect number of faces')
            self._vertex_values = None
            self._vertex_values_indexed_by_faces = values
        else:
            raise ValueError('indexed must be None or "faces"')

    def get_face_colors(self, indexed=None):
        if False:
            return 10
        "Get the face colors\n\n        Parameters\n        ----------\n        indexed : str | None\n            If indexed is None, return (Nf, 4) array of face colors.\n            If indexed=='faces', then instead return an indexed array\n            (Nf, 3, 4)  (note this is just the same array with each color\n            repeated three times).\n\n        Returns\n        -------\n        colors : ndarray\n            The colors.\n        "
        if indexed is None:
            return self._face_colors
        elif indexed == 'faces':
            if self._face_colors_indexed_by_faces is None and self._face_colors is not None:
                Nf = self._face_colors.shape[0]
                self._face_colors_indexed_by_faces = np.empty((Nf, 3, 4), dtype=self._face_colors.dtype)
                self._face_colors_indexed_by_faces[:] = self._face_colors.reshape(Nf, 1, 4)
            return self._face_colors_indexed_by_faces
        else:
            raise ValueError("Invalid indexing mode. Accepts: None, 'faces'")

    def set_face_colors(self, colors, indexed=None):
        if False:
            print('Hello World!')
        "Set the face color array\n\n        Parameters\n        ----------\n        colors : array\n            Array of colors. Must have shape (Nf, 4) (indexed by face),\n            or shape (Nf, 3, 4) (face colors indexed by faces).\n        indexed : str | None\n            Should be 'faces' if colors are indexed by faces.\n        "
        colors = _fix_colors(colors)
        if colors.shape[0] != self.n_faces:
            raise ValueError('incorrect number of colors %s, expected %s' % (colors.shape[0], self.n_faces))
        if indexed is None:
            if colors.ndim != 2:
                raise ValueError('colors must be 2D if indexed is None')
            self._face_colors = colors
            self._face_colors_indexed_by_faces = None
        elif indexed == 'faces':
            if colors.ndim != 3:
                raise ValueError('colors must be 3D if indexed is "faces"')
            self._face_colors = None
            self._face_colors_indexed_by_faces = colors
        else:
            raise ValueError('indexed must be None or "faces"')

    @property
    def n_faces(self):
        if False:
            print('Hello World!')
        'The number of faces in the mesh'
        if self._faces is not None:
            return self._faces.shape[0]
        elif self._vertices_indexed_by_faces is not None:
            return self._vertices_indexed_by_faces.shape[0]

    @property
    def n_vertices(self):
        if False:
            i = 10
            return i + 15
        'The number of vertices in the mesh'
        if self._vertices is None:
            self._compute_unindexed_vertices()
        return len(self._vertices)

    def get_edge_colors(self):
        if False:
            return 10
        return self._edge_colors

    def _compute_unindexed_vertices(self):
        if False:
            while True:
                i = 10
        faces = self._vertices_indexed_by_faces
        verts = {}
        self._faces = np.empty(faces.shape[:2], dtype=np.uint32)
        self._vertices = []
        self._vertex_faces = []
        self._face_normals = None
        self._vertex_normals = None
        for i in range(faces.shape[0]):
            face = faces[i]
            for j in range(face.shape[0]):
                pt = face[j]
                pt2 = tuple([round(x * 100000000000000.0) for x in pt])
                index = verts.get(pt2, None)
                if index is None:
                    self._vertices.append(pt)
                    self._vertex_faces.append([])
                    index = len(self._vertices) - 1
                    verts[pt2] = index
                self._vertex_faces[index].append(i)
                self._faces[i, j] = index
        self._vertices = np.array(self._vertices, dtype=np.float32)

    def get_vertex_faces(self):
        if False:
            return 10
        'List mapping each vertex index to a list of face indices that use it.'
        if self._vertex_faces is None:
            self._vertex_faces = [[] for i in range(len(self.get_vertices()))]
            for i in range(self._faces.shape[0]):
                face = self._faces[i]
                for ind in face:
                    self._vertex_faces[ind].append(i)
        return self._vertex_faces

    def _compute_edges(self, indexed=None):
        if False:
            print('Hello World!')
        if indexed is None:
            if self._faces is not None:
                nf = len(self._faces)
                edges = np.empty(nf * 3, dtype=[('i', np.uint32, 2)])
                edges['i'][0:nf] = self._faces[:, :2]
                edges['i'][nf:2 * nf] = self._faces[:, 1:3]
                edges['i'][-nf:, 0] = self._faces[:, 2]
                edges['i'][-nf:, 1] = self._faces[:, 0]
                mask = edges['i'][:, 0] > edges['i'][:, 1]
                edges['i'][mask] = edges['i'][mask][:, ::-1]
                self._edges = np.unique(edges)['i']
            else:
                raise Exception('MeshData cannot generate edges--no faces in this data.')
        elif indexed == 'faces':
            if self._vertices_indexed_by_faces is not None:
                verts = self._vertices_indexed_by_faces
                edges = np.empty((verts.shape[0], 3, 2), dtype=np.uint32)
                nf = verts.shape[0]
                edges[:, 0, 0] = np.arange(nf) * 3
                edges[:, 0, 1] = edges[:, 0, 0] + 1
                edges[:, 1, 0] = edges[:, 0, 1]
                edges[:, 1, 1] = edges[:, 1, 0] + 1
                edges[:, 2, 0] = edges[:, 1, 1]
                edges[:, 2, 1] = edges[:, 0, 0]
                self._edges_indexed_by_faces = edges
            else:
                raise Exception('MeshData cannot generate edges--no faces in this data.')
        else:
            raise ValueError("Invalid indexing mode. Accepts: None, 'faces'")

    def save(self):
        if False:
            for i in range(10):
                print('nop')
        'Serialize this mesh to a string appropriate for disk storage\n\n        Returns\n        -------\n        state : dict\n            The state.\n        '
        import pickle
        if self._faces is not None:
            names = ['_vertices', '_faces']
        else:
            names = ['_vertices_indexed_by_faces']
        if self._vertex_colors is not None:
            names.append('_vertex_colors')
        elif self._vertex_colors_indexed_by_faces is not None:
            names.append('_vertex_colors_indexed_by_faces')
        if self._face_colors is not None:
            names.append('_face_colors')
        elif self._face_colors_indexed_by_faces is not None:
            names.append('_face_colors_indexed_by_faces')
        state = dict([(n, getattr(self, n)) for n in names])
        return pickle.dumps(state)

    def restore(self, state):
        if False:
            while True:
                i = 10
        'Restore the state of a mesh previously saved using save()\n\n        Parameters\n        ----------\n        state : dict\n            The previous state.\n        '
        import pickle
        state = pickle.loads(state)
        for k in state:
            if isinstance(state[k], list):
                state[k] = np.array(state[k])
            setattr(self, k, state[k])

    def is_empty(self):
        if False:
            print('Hello World!')
        'Check if any vertices or faces are defined.'
        return self._faces is None