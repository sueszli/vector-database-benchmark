"""
Antigrain Geometry Point Collection

This collection provides fast points. Output quality is perfect.
"""
from ... import glsl
from .raw_point_collection import RawPointCollection

class AggPointCollection(RawPointCollection):
    """
    Antigrain Geometry Point Collection

    This collection provides fast points. Output quality is perfect.
    """

    def __init__(self, user_dtype=None, transform=None, vertex=None, fragment=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Initialize the collection.\n\n        Parameters\n        ----------\n        user_dtype: list\n            The base dtype can be completed (appended) by the used_dtype. It\n            only make sense if user also provide vertex and/or fragment shaders\n\n        vertex: string\n            Vertex shader code\n\n        fragment: string\n            Fragment  shader code\n\n        transform : Transform instance\n            Used to define the GLSL transform(vec4) function\n\n        color : string\n            'local', 'shared' or 'global'\n        "
        if vertex is None:
            vertex = glsl.get('collections/agg-point.vert')
        if fragment is None:
            fragment = glsl.get('collections/agg-point.frag')
        RawPointCollection.__init__(self, user_dtype=user_dtype, transform=transform, vertex=vertex, fragment=fragment, **kwargs)