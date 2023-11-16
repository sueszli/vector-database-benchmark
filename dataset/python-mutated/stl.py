import numpy as np

class HeaderError(Exception):
    pass
_stl_dtype = np.dtype([('normals', np.float32, 3), ('vertices', np.float32, (3, 3)), ('attributes', np.uint16)])
_stl_dtype_header = np.dtype([('header', np.void, 80), ('face_count', np.int32)])

def load_stl(file_obj, file_type=None):
    if False:
        print('Hello World!')
    '\n    Load an STL file from a file object.\n\n    Parameters\n    ----------\n    file_obj: open file- like object\n    file_type: not used\n\n    Returns\n    -------\n    loaded: kwargs for a Trimesh constructor with keys:\n              vertices:     (n,3) float, vertices\n              faces:        (m,3) int, indexes of vertices\n              face_normals: (m,3) float, normal vector of each face\n    '
    file_pos = file_obj.tell()
    try:
        return load_stl_binary(file_obj)
    except HeaderError:
        file_obj.seek(file_pos)
        return load_stl_ascii(file_obj)

def load_stl_binary(file_obj):
    if False:
        while True:
            i = 10
    '\n    Load a binary STL file from a file object.\n\n    Parameters\n    ----------\n    file_obj: open file- like object\n\n    Returns\n    -------\n    loaded: kwargs for a Trimesh constructor with keys:\n              vertices:     (n,3) float, vertices\n              faces:        (m,3) int, indexes of vertices\n              face_normals: (m,3) float, normal vector of each face\n    '
    header_length = _stl_dtype_header.itemsize
    header_data = file_obj.read(header_length)
    if len(header_data) < header_length:
        raise HeaderError('Binary STL file not long enough to contain header!')
    header = np.fromstring(header_data, dtype=_stl_dtype_header)
    data_start = file_obj.tell()
    file_obj.seek(0, 2)
    data_end = file_obj.tell()
    file_obj.seek(data_start)
    len_data = data_end - data_start
    len_expected = header['face_count'] * _stl_dtype.itemsize
    if len_data != len_expected:
        raise HeaderError('Binary STL has incorrect length in header!')
    faces = np.arange(header['face_count'] * 3).reshape((-1, 3))
    blob = np.fromstring(file_obj.read(), dtype=_stl_dtype)
    result = {'vertices': blob['vertices'].reshape((-1, 3)), 'face_normals': blob['normals'].reshape((-1, 3)), 'faces': faces}
    return result

def load_stl_ascii(file_obj):
    if False:
        return 10
    '\n    Load an ASCII STL file from a file object.\n\n    Parameters\n    ----------\n    file_obj: open file- like object\n\n    Returns\n    -------\n    loaded: kwargs for a Trimesh constructor with keys:\n              vertices:     (n,3) float, vertices\n              faces:        (m,3) int, indexes of vertices\n              face_normals: (m,3) float, normal vector of each face\n    '
    file_obj.readline()
    text = file_obj.read()
    if hasattr(text, 'decode'):
        text = text.decode('utf-8')
    text = text.lower().split('endsolid')[0]
    blob = np.array(text.split())
    face_len = 21
    face_count = len(blob) / face_len
    if len(blob) % face_len != 0:
        raise HeaderError('Incorrect number of values in STL file!')
    face_count = int(face_count)
    offset = face_len * np.arange(face_count).reshape((-1, 1))
    normal_index = np.tile([2, 3, 4], (face_count, 1)) + offset
    vertex_index = np.tile([8, 9, 10, 12, 13, 14, 16, 17, 18], (face_count, 1)) + offset
    faces = np.arange(face_count * 3).reshape((-1, 3))
    face_normals = blob[normal_index].astype(np.float64)
    vertices = blob[vertex_index.reshape((-1, 3))].astype(np.float64)
    return {'vertices': vertices, 'faces': faces, 'face_normals': face_normals}
_stl_loaders = {'stl': load_stl, 'stl_ascii': load_stl}