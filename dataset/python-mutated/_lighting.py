from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class Lighting(_BaseTraceHierarchyType):
    _parent_path_str = 'surface'
    _path_str = 'surface.lighting'
    _valid_props = {'ambient', 'diffuse', 'fresnel', 'roughness', 'specular'}

    @property
    def ambient(self):
        if False:
            i = 10
            return i + 15
        "\n        Ambient light increases overall color visibility but can wash\n        out the image.\n\n        The 'ambient' property is a number and may be specified as:\n          - An int or float in the interval [0, 1]\n\n        Returns\n        -------\n        int|float\n        "
        return self['ambient']

    @ambient.setter
    def ambient(self, val):
        if False:
            i = 10
            return i + 15
        self['ambient'] = val

    @property
    def diffuse(self):
        if False:
            while True:
                i = 10
        "\n        Represents the extent that incident rays are reflected in a\n        range of angles.\n\n        The 'diffuse' property is a number and may be specified as:\n          - An int or float in the interval [0, 1]\n\n        Returns\n        -------\n        int|float\n        "
        return self['diffuse']

    @diffuse.setter
    def diffuse(self, val):
        if False:
            print('Hello World!')
        self['diffuse'] = val

    @property
    def fresnel(self):
        if False:
            while True:
                i = 10
        "\n        Represents the reflectance as a dependency of the viewing\n        angle; e.g. paper is reflective when viewing it from the edge\n        of the paper (almost 90 degrees), causing shine.\n\n        The 'fresnel' property is a number and may be specified as:\n          - An int or float in the interval [0, 5]\n\n        Returns\n        -------\n        int|float\n        "
        return self['fresnel']

    @fresnel.setter
    def fresnel(self, val):
        if False:
            return 10
        self['fresnel'] = val

    @property
    def roughness(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Alters specular reflection; the rougher the surface, the wider\n        and less contrasty the shine.\n\n        The 'roughness' property is a number and may be specified as:\n          - An int or float in the interval [0, 1]\n\n        Returns\n        -------\n        int|float\n        "
        return self['roughness']

    @roughness.setter
    def roughness(self, val):
        if False:
            print('Hello World!')
        self['roughness'] = val

    @property
    def specular(self):
        if False:
            return 10
        "\n        Represents the level that incident rays are reflected in a\n        single direction, causing shine.\n\n        The 'specular' property is a number and may be specified as:\n          - An int or float in the interval [0, 2]\n\n        Returns\n        -------\n        int|float\n        "
        return self['specular']

    @specular.setter
    def specular(self, val):
        if False:
            print('Hello World!')
        self['specular'] = val

    @property
    def _prop_descriptions(self):
        if False:
            for i in range(10):
                print('nop')
        return '        ambient\n            Ambient light increases overall color visibility but\n            can wash out the image.\n        diffuse\n            Represents the extent that incident rays are reflected\n            in a range of angles.\n        fresnel\n            Represents the reflectance as a dependency of the\n            viewing angle; e.g. paper is reflective when viewing it\n            from the edge of the paper (almost 90 degrees), causing\n            shine.\n        roughness\n            Alters specular reflection; the rougher the surface,\n            the wider and less contrasty the shine.\n        specular\n            Represents the level that incident rays are reflected\n            in a single direction, causing shine.\n        '

    def __init__(self, arg=None, ambient=None, diffuse=None, fresnel=None, roughness=None, specular=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Construct a new Lighting object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.surface.Lighting`\n        ambient\n            Ambient light increases overall color visibility but\n            can wash out the image.\n        diffuse\n            Represents the extent that incident rays are reflected\n            in a range of angles.\n        fresnel\n            Represents the reflectance as a dependency of the\n            viewing angle; e.g. paper is reflective when viewing it\n            from the edge of the paper (almost 90 degrees), causing\n            shine.\n        roughness\n            Alters specular reflection; the rougher the surface,\n            the wider and less contrasty the shine.\n        specular\n            Represents the level that incident rays are reflected\n            in a single direction, causing shine.\n\n        Returns\n        -------\n        Lighting\n        '
        super(Lighting, self).__init__('lighting')
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
            raise ValueError('The first argument to the plotly.graph_objs.surface.Lighting\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.surface.Lighting`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('ambient', None)
        _v = ambient if ambient is not None else _v
        if _v is not None:
            self['ambient'] = _v
        _v = arg.pop('diffuse', None)
        _v = diffuse if diffuse is not None else _v
        if _v is not None:
            self['diffuse'] = _v
        _v = arg.pop('fresnel', None)
        _v = fresnel if fresnel is not None else _v
        if _v is not None:
            self['fresnel'] = _v
        _v = arg.pop('roughness', None)
        _v = roughness if roughness is not None else _v
        if _v is not None:
            self['roughness'] = _v
        _v = arg.pop('specular', None)
        _v = specular if specular is not None else _v
        if _v is not None:
            self['specular'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False