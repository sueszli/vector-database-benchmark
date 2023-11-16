from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Camera(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout.scene'
    _path_str = 'layout.scene.camera'
    _valid_props = {'center', 'eye', 'projection', 'up'}

    @property
    def center(self):
        if False:
            print('Hello World!')
        "\n        Sets the (x,y,z) components of the 'center' camera vector This\n        vector determines the translation (x,y,z) space about the\n        center of this scene. By default, there is no such translation.\n\n        The 'center' property is an instance of Center\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.scene.camera.Center`\n          - A dict of string/value properties that will be passed\n            to the Center constructor\n\n            Supported dict properties:\n\n                x\n\n                y\n\n                z\n\n        Returns\n        -------\n        plotly.graph_objs.layout.scene.camera.Center\n        "
        return self['center']

    @center.setter
    def center(self, val):
        if False:
            i = 10
            return i + 15
        self['center'] = val

    @property
    def eye(self):
        if False:
            print('Hello World!')
        "\n        Sets the (x,y,z) components of the 'eye' camera vector. This\n        vector determines the view point about the origin of this\n        scene.\n\n        The 'eye' property is an instance of Eye\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.scene.camera.Eye`\n          - A dict of string/value properties that will be passed\n            to the Eye constructor\n\n            Supported dict properties:\n\n                x\n\n                y\n\n                z\n\n        Returns\n        -------\n        plotly.graph_objs.layout.scene.camera.Eye\n        "
        return self['eye']

    @eye.setter
    def eye(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['eye'] = val

    @property
    def projection(self):
        if False:
            i = 10
            return i + 15
        '\n        The \'projection\' property is an instance of Projection\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.scene.camera.Projection`\n          - A dict of string/value properties that will be passed\n            to the Projection constructor\n\n            Supported dict properties:\n\n                type\n                    Sets the projection type. The projection type\n                    could be either "perspective" or\n                    "orthographic". The default is "perspective".\n\n        Returns\n        -------\n        plotly.graph_objs.layout.scene.camera.Projection\n        '
        return self['projection']

    @projection.setter
    def projection(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['projection'] = val

    @property
    def up(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the (x,y,z) components of the 'up' camera vector. This\n        vector determines the up direction of this scene with respect\n        to the page. The default is *{x: 0, y: 0, z: 1}* which means\n        that the z axis points up.\n\n        The 'up' property is an instance of Up\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.scene.camera.Up`\n          - A dict of string/value properties that will be passed\n            to the Up constructor\n\n            Supported dict properties:\n\n                x\n\n                y\n\n                z\n\n        Returns\n        -------\n        plotly.graph_objs.layout.scene.camera.Up\n        "
        return self['up']

    @up.setter
    def up(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['up'] = val

    @property
    def _prop_descriptions(self):
        if False:
            while True:
                i = 10
        return "        center\n            Sets the (x,y,z) components of the 'center' camera\n            vector This vector determines the translation (x,y,z)\n            space about the center of this scene. By default, there\n            is no such translation.\n        eye\n            Sets the (x,y,z) components of the 'eye' camera vector.\n            This vector determines the view point about the origin\n            of this scene.\n        projection\n            :class:`plotly.graph_objects.layout.scene.camera.Projec\n            tion` instance or dict with compatible properties\n        up\n            Sets the (x,y,z) components of the 'up' camera vector.\n            This vector determines the up direction of this scene\n            with respect to the page. The default is *{x: 0, y: 0,\n            z: 1}* which means that the z axis points up.\n        "

    def __init__(self, arg=None, center=None, eye=None, projection=None, up=None, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Construct a new Camera object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.layout.scene.Camera`\n        center\n            Sets the (x,y,z) components of the 'center' camera\n            vector This vector determines the translation (x,y,z)\n            space about the center of this scene. By default, there\n            is no such translation.\n        eye\n            Sets the (x,y,z) components of the 'eye' camera vector.\n            This vector determines the view point about the origin\n            of this scene.\n        projection\n            :class:`plotly.graph_objects.layout.scene.camera.Projec\n            tion` instance or dict with compatible properties\n        up\n            Sets the (x,y,z) components of the 'up' camera vector.\n            This vector determines the up direction of this scene\n            with respect to the page. The default is *{x: 0, y: 0,\n            z: 1}* which means that the z axis points up.\n\n        Returns\n        -------\n        Camera\n        "
        super(Camera, self).__init__('camera')
        if '_parent' in kwargs:
            self._parent = kwargs['_parent']
            return
        if arg is None:
            arg = {}
        elif isinstance(arg, self.__class__):
            arg = arg.to_plotly_json()
        elif isinstance(arg, dict):
            arg = _copy.copy(arg)
        else:
            raise ValueError('The first argument to the plotly.graph_objs.layout.scene.Camera\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.scene.Camera`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('center', None)
        _v = center if center is not None else _v
        if _v is not None:
            self['center'] = _v
        _v = arg.pop('eye', None)
        _v = eye if eye is not None else _v
        if _v is not None:
            self['eye'] = _v
        _v = arg.pop('projection', None)
        _v = projection if projection is not None else _v
        if _v is not None:
            self['projection'] = _v
        _v = arg.pop('up', None)
        _v = up if up is not None else _v
        if _v is not None:
            self['up'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False