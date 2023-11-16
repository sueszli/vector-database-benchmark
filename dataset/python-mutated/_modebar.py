from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Modebar(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout'
    _path_str = 'layout.modebar'
    _valid_props = {'activecolor', 'add', 'addsrc', 'bgcolor', 'color', 'orientation', 'remove', 'removesrc', 'uirevision'}

    @property
    def activecolor(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the color of the active or hovered on icons in the\n        modebar.\n\n        The 'activecolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['activecolor']

    @activecolor.setter
    def activecolor(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['activecolor'] = val

    @property
    def add(self):
        if False:
            while True:
                i = 10
        '\n        Determines which predefined modebar buttons to add. Please note\n        that these buttons will only be shown if they are compatible\n        with all trace types used in a graph. Similar to\n        `config.modeBarButtonsToAdd` option. This may include\n        "v1hovermode", "hoverclosest", "hovercompare", "togglehover",\n        "togglespikelines", "drawline", "drawopenpath",\n        "drawclosedpath", "drawcircle", "drawrect", "eraseshape".\n\n        The \'add\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        str|numpy.ndarray\n        '
        return self['add']

    @add.setter
    def add(self, val):
        if False:
            print('Hello World!')
        self['add'] = val

    @property
    def addsrc(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the source reference on Chart Studio Cloud for `add`.\n\n        The 'addsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['addsrc']

    @addsrc.setter
    def addsrc(self, val):
        if False:
            while True:
                i = 10
        self['addsrc'] = val

    @property
    def bgcolor(self):
        if False:
            return 10
        "\n        Sets the background color of the modebar.\n\n        The 'bgcolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['bgcolor']

    @bgcolor.setter
    def bgcolor(self, val):
        if False:
            print('Hello World!')
        self['bgcolor'] = val

    @property
    def color(self):
        if False:
            print('Hello World!')
        "\n        Sets the color of the icons in the modebar.\n\n        The 'color' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['color']

    @color.setter
    def color(self, val):
        if False:
            print('Hello World!')
        self['color'] = val

    @property
    def orientation(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the orientation of the modebar.\n\n        The 'orientation' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['v', 'h']\n\n        Returns\n        -------\n        Any\n        "
        return self['orientation']

    @orientation.setter
    def orientation(self, val):
        if False:
            while True:
                i = 10
        self['orientation'] = val

    @property
    def remove(self):
        if False:
            return 10
        '\n        Determines which predefined modebar buttons to remove. Similar\n        to `config.modeBarButtonsToRemove` option. This may include\n        "autoScale2d", "autoscale", "editInChartStudio",\n        "editinchartstudio", "hoverCompareCartesian", "hovercompare",\n        "lasso", "lasso2d", "orbitRotation", "orbitrotation", "pan",\n        "pan2d", "pan3d", "reset", "resetCameraDefault3d",\n        "resetCameraLastSave3d", "resetGeo", "resetSankeyGroup",\n        "resetScale2d", "resetViewMapbox", "resetViews",\n        "resetcameradefault", "resetcameralastsave",\n        "resetsankeygroup", "resetscale", "resetview", "resetviews",\n        "select", "select2d", "sendDataToCloud", "senddatatocloud",\n        "tableRotation", "tablerotation", "toImage", "toggleHover",\n        "toggleSpikelines", "togglehover", "togglespikelines",\n        "toimage", "zoom", "zoom2d", "zoom3d", "zoomIn2d", "zoomInGeo",\n        "zoomInMapbox", "zoomOut2d", "zoomOutGeo", "zoomOutMapbox",\n        "zoomin", "zoomout".\n\n        The \'remove\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        str|numpy.ndarray\n        '
        return self['remove']

    @remove.setter
    def remove(self, val):
        if False:
            return 10
        self['remove'] = val

    @property
    def removesrc(self):
        if False:
            while True:
                i = 10
        "\n        Sets the source reference on Chart Studio Cloud for `remove`.\n\n        The 'removesrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['removesrc']

    @removesrc.setter
    def removesrc(self, val):
        if False:
            print('Hello World!')
        self['removesrc'] = val

    @property
    def uirevision(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Controls persistence of user-driven changes related to the\n        modebar, including `hovermode`, `dragmode`, and `showspikes` at\n        both the root level and inside subplots. Defaults to\n        `layout.uirevision`.\n\n        The 'uirevision' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['uirevision']

    @uirevision.setter
    def uirevision(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['uirevision'] = val

    @property
    def _prop_descriptions(self):
        if False:
            for i in range(10):
                print('nop')
        return '        activecolor\n            Sets the color of the active or hovered on icons in the\n            modebar.\n        add\n            Determines which predefined modebar buttons to add.\n            Please note that these buttons will only be shown if\n            they are compatible with all trace types used in a\n            graph. Similar to `config.modeBarButtonsToAdd` option.\n            This may include "v1hovermode", "hoverclosest",\n            "hovercompare", "togglehover", "togglespikelines",\n            "drawline", "drawopenpath", "drawclosedpath",\n            "drawcircle", "drawrect", "eraseshape".\n        addsrc\n            Sets the source reference on Chart Studio Cloud for\n            `add`.\n        bgcolor\n            Sets the background color of the modebar.\n        color\n            Sets the color of the icons in the modebar.\n        orientation\n            Sets the orientation of the modebar.\n        remove\n            Determines which predefined modebar buttons to remove.\n            Similar to `config.modeBarButtonsToRemove` option. This\n            may include "autoScale2d", "autoscale",\n            "editInChartStudio", "editinchartstudio",\n            "hoverCompareCartesian", "hovercompare", "lasso",\n            "lasso2d", "orbitRotation", "orbitrotation", "pan",\n            "pan2d", "pan3d", "reset", "resetCameraDefault3d",\n            "resetCameraLastSave3d", "resetGeo",\n            "resetSankeyGroup", "resetScale2d", "resetViewMapbox",\n            "resetViews", "resetcameradefault",\n            "resetcameralastsave", "resetsankeygroup",\n            "resetscale", "resetview", "resetviews", "select",\n            "select2d", "sendDataToCloud", "senddatatocloud",\n            "tableRotation", "tablerotation", "toImage",\n            "toggleHover", "toggleSpikelines", "togglehover",\n            "togglespikelines", "toimage", "zoom", "zoom2d",\n            "zoom3d", "zoomIn2d", "zoomInGeo", "zoomInMapbox",\n            "zoomOut2d", "zoomOutGeo", "zoomOutMapbox", "zoomin",\n            "zoomout".\n        removesrc\n            Sets the source reference on Chart Studio Cloud for\n            `remove`.\n        uirevision\n            Controls persistence of user-driven changes related to\n            the modebar, including `hovermode`, `dragmode`, and\n            `showspikes` at both the root level and inside\n            subplots. Defaults to `layout.uirevision`.\n        '

    def __init__(self, arg=None, activecolor=None, add=None, addsrc=None, bgcolor=None, color=None, orientation=None, remove=None, removesrc=None, uirevision=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a new Modebar object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.layout.Modebar`\n        activecolor\n            Sets the color of the active or hovered on icons in the\n            modebar.\n        add\n            Determines which predefined modebar buttons to add.\n            Please note that these buttons will only be shown if\n            they are compatible with all trace types used in a\n            graph. Similar to `config.modeBarButtonsToAdd` option.\n            This may include "v1hovermode", "hoverclosest",\n            "hovercompare", "togglehover", "togglespikelines",\n            "drawline", "drawopenpath", "drawclosedpath",\n            "drawcircle", "drawrect", "eraseshape".\n        addsrc\n            Sets the source reference on Chart Studio Cloud for\n            `add`.\n        bgcolor\n            Sets the background color of the modebar.\n        color\n            Sets the color of the icons in the modebar.\n        orientation\n            Sets the orientation of the modebar.\n        remove\n            Determines which predefined modebar buttons to remove.\n            Similar to `config.modeBarButtonsToRemove` option. This\n            may include "autoScale2d", "autoscale",\n            "editInChartStudio", "editinchartstudio",\n            "hoverCompareCartesian", "hovercompare", "lasso",\n            "lasso2d", "orbitRotation", "orbitrotation", "pan",\n            "pan2d", "pan3d", "reset", "resetCameraDefault3d",\n            "resetCameraLastSave3d", "resetGeo",\n            "resetSankeyGroup", "resetScale2d", "resetViewMapbox",\n            "resetViews", "resetcameradefault",\n            "resetcameralastsave", "resetsankeygroup",\n            "resetscale", "resetview", "resetviews", "select",\n            "select2d", "sendDataToCloud", "senddatatocloud",\n            "tableRotation", "tablerotation", "toImage",\n            "toggleHover", "toggleSpikelines", "togglehover",\n            "togglespikelines", "toimage", "zoom", "zoom2d",\n            "zoom3d", "zoomIn2d", "zoomInGeo", "zoomInMapbox",\n            "zoomOut2d", "zoomOutGeo", "zoomOutMapbox", "zoomin",\n            "zoomout".\n        removesrc\n            Sets the source reference on Chart Studio Cloud for\n            `remove`.\n        uirevision\n            Controls persistence of user-driven changes related to\n            the modebar, including `hovermode`, `dragmode`, and\n            `showspikes` at both the root level and inside\n            subplots. Defaults to `layout.uirevision`.\n\n        Returns\n        -------\n        Modebar\n        '
        super(Modebar, self).__init__('modebar')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.Modebar\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.Modebar`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('activecolor', None)
        _v = activecolor if activecolor is not None else _v
        if _v is not None:
            self['activecolor'] = _v
        _v = arg.pop('add', None)
        _v = add if add is not None else _v
        if _v is not None:
            self['add'] = _v
        _v = arg.pop('addsrc', None)
        _v = addsrc if addsrc is not None else _v
        if _v is not None:
            self['addsrc'] = _v
        _v = arg.pop('bgcolor', None)
        _v = bgcolor if bgcolor is not None else _v
        if _v is not None:
            self['bgcolor'] = _v
        _v = arg.pop('color', None)
        _v = color if color is not None else _v
        if _v is not None:
            self['color'] = _v
        _v = arg.pop('orientation', None)
        _v = orientation if orientation is not None else _v
        if _v is not None:
            self['orientation'] = _v
        _v = arg.pop('remove', None)
        _v = remove if remove is not None else _v
        if _v is not None:
            self['remove'] = _v
        _v = arg.pop('removesrc', None)
        _v = removesrc if removesrc is not None else _v
        if _v is not None:
            self['removesrc'] = _v
        _v = arg.pop('uirevision', None)
        _v = uirevision if uirevision is not None else _v
        if _v is not None:
            self['uirevision'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False