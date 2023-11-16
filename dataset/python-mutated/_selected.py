from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class Selected(_BaseTraceHierarchyType):
    _parent_path_str = 'scatterpolar'
    _path_str = 'scatterpolar.selected'
    _valid_props = {'marker', 'textfont'}

    @property
    def marker(self):
        if False:
            while True:
                i = 10
        "\n        The 'marker' property is an instance of Marker\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.scatterpolar.selected.Marker`\n          - A dict of string/value properties that will be passed\n            to the Marker constructor\n\n            Supported dict properties:\n\n                color\n                    Sets the marker color of selected points.\n                opacity\n                    Sets the marker opacity of selected points.\n                size\n                    Sets the marker size of selected points.\n\n        Returns\n        -------\n        plotly.graph_objs.scatterpolar.selected.Marker\n        "
        return self['marker']

    @marker.setter
    def marker(self, val):
        if False:
            return 10
        self['marker'] = val

    @property
    def textfont(self):
        if False:
            while True:
                i = 10
        "\n        The 'textfont' property is an instance of Textfont\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.scatterpolar.selected.Textfont`\n          - A dict of string/value properties that will be passed\n            to the Textfont constructor\n\n            Supported dict properties:\n\n                color\n                    Sets the text font color of selected points.\n\n        Returns\n        -------\n        plotly.graph_objs.scatterpolar.selected.Textfont\n        "
        return self['textfont']

    @textfont.setter
    def textfont(self, val):
        if False:
            print('Hello World!')
        self['textfont'] = val

    @property
    def _prop_descriptions(self):
        if False:
            while True:
                i = 10
        return '        marker\n            :class:`plotly.graph_objects.scatterpolar.selected.Mark\n            er` instance or dict with compatible properties\n        textfont\n            :class:`plotly.graph_objects.scatterpolar.selected.Text\n            font` instance or dict with compatible properties\n        '

    def __init__(self, arg=None, marker=None, textfont=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a new Selected object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.scatterpolar.Selected`\n        marker\n            :class:`plotly.graph_objects.scatterpolar.selected.Mark\n            er` instance or dict with compatible properties\n        textfont\n            :class:`plotly.graph_objects.scatterpolar.selected.Text\n            font` instance or dict with compatible properties\n\n        Returns\n        -------\n        Selected\n        '
        super(Selected, self).__init__('selected')
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
            raise ValueError('The first argument to the plotly.graph_objs.scatterpolar.Selected\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.scatterpolar.Selected`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('marker', None)
        _v = marker if marker is not None else _v
        if _v is not None:
            self['marker'] = _v
        _v = arg.pop('textfont', None)
        _v = textfont if textfont is not None else _v
        if _v is not None:
            self['textfont'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False