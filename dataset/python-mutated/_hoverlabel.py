from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class Hoverlabel(_BaseTraceHierarchyType):
    _parent_path_str = 'pointcloud'
    _path_str = 'pointcloud.hoverlabel'
    _valid_props = {'align', 'alignsrc', 'bgcolor', 'bgcolorsrc', 'bordercolor', 'bordercolorsrc', 'font', 'namelength', 'namelengthsrc'}

    @property
    def align(self):
        if False:
            return 10
        "\n        Sets the horizontal alignment of the text content within hover\n        label box. Has an effect only if the hover label text spans\n        more two or more lines\n\n        The 'align' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['left', 'right', 'auto']\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        Any|numpy.ndarray\n        "
        return self['align']

    @align.setter
    def align(self, val):
        if False:
            return 10
        self['align'] = val

    @property
    def alignsrc(self):
        if False:
            return 10
        "\n        Sets the source reference on Chart Studio Cloud for `align`.\n\n        The 'alignsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['alignsrc']

    @alignsrc.setter
    def alignsrc(self, val):
        if False:
            i = 10
            return i + 15
        self['alignsrc'] = val

    @property
    def bgcolor(self):
        if False:
            print('Hello World!')
        "\n        Sets the background color of the hover labels for this trace\n\n        The 'bgcolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n          - A list or array of any of the above\n\n        Returns\n        -------\n        str|numpy.ndarray\n        "
        return self['bgcolor']

    @bgcolor.setter
    def bgcolor(self, val):
        if False:
            return 10
        self['bgcolor'] = val

    @property
    def bgcolorsrc(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the source reference on Chart Studio Cloud for `bgcolor`.\n\n        The 'bgcolorsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['bgcolorsrc']

    @bgcolorsrc.setter
    def bgcolorsrc(self, val):
        if False:
            return 10
        self['bgcolorsrc'] = val

    @property
    def bordercolor(self):
        if False:
            return 10
        "\n        Sets the border color of the hover labels for this trace.\n\n        The 'bordercolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n          - A list or array of any of the above\n\n        Returns\n        -------\n        str|numpy.ndarray\n        "
        return self['bordercolor']

    @bordercolor.setter
    def bordercolor(self, val):
        if False:
            while True:
                i = 10
        self['bordercolor'] = val

    @property
    def bordercolorsrc(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the source reference on Chart Studio Cloud for\n        `bordercolor`.\n\n        The 'bordercolorsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['bordercolorsrc']

    @bordercolorsrc.setter
    def bordercolorsrc(self, val):
        if False:
            while True:
                i = 10
        self['bordercolorsrc'] = val

    @property
    def font(self):
        if False:
            i = 10
            return i + 15
        '\n        Sets the font used in hover labels.\n\n        The \'font\' property is an instance of Font\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.pointcloud.hoverlabel.Font`\n          - A dict of string/value properties that will be passed\n            to the Font constructor\n\n            Supported dict properties:\n\n                color\n\n                colorsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `color`.\n                family\n                    HTML font family - the typeface that will be\n                    applied by the web browser. The web browser\n                    will only be able to apply a font if it is\n                    available on the system which it operates.\n                    Provide multiple font families, separated by\n                    commas, to indicate the preference in which to\n                    apply fonts if they aren\'t available on the\n                    system. The Chart Studio Cloud (at\n                    https://chart-studio.plotly.com or on-premise)\n                    generates images on a server, where only a\n                    select number of fonts are installed and\n                    supported. These include "Arial", "Balto",\n                    "Courier New", "Droid Sans",, "Droid Serif",\n                    "Droid Sans Mono", "Gravitas One", "Old\n                    Standard TT", "Open Sans", "Overpass", "PT Sans\n                    Narrow", "Raleway", "Times New Roman".\n                familysrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `family`.\n                size\n\n                sizesrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `size`.\n\n        Returns\n        -------\n        plotly.graph_objs.pointcloud.hoverlabel.Font\n        '
        return self['font']

    @font.setter
    def font(self, val):
        if False:
            return 10
        self['font'] = val

    @property
    def namelength(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the default length (in number of characters) of the trace\n        name in the hover labels for all traces. -1 shows the whole\n        name regardless of length. 0-3 shows the first 0-3 characters,\n        and an integer >3 will show the whole name if it is less than\n        that many characters, but if it is longer, will truncate to\n        `namelength - 3` characters and add an ellipsis.\n\n        The 'namelength' property is a integer and may be specified as:\n          - An int (or float that will be cast to an int)\n            in the interval [-1, 9223372036854775807]\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        int|numpy.ndarray\n        "
        return self['namelength']

    @namelength.setter
    def namelength(self, val):
        if False:
            print('Hello World!')
        self['namelength'] = val

    @property
    def namelengthsrc(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the source reference on Chart Studio Cloud for\n        `namelength`.\n\n        The 'namelengthsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['namelengthsrc']

    @namelengthsrc.setter
    def namelengthsrc(self, val):
        if False:
            while True:
                i = 10
        self['namelengthsrc'] = val

    @property
    def _prop_descriptions(self):
        if False:
            for i in range(10):
                print('nop')
        return '        align\n            Sets the horizontal alignment of the text content\n            within hover label box. Has an effect only if the hover\n            label text spans more two or more lines\n        alignsrc\n            Sets the source reference on Chart Studio Cloud for\n            `align`.\n        bgcolor\n            Sets the background color of the hover labels for this\n            trace\n        bgcolorsrc\n            Sets the source reference on Chart Studio Cloud for\n            `bgcolor`.\n        bordercolor\n            Sets the border color of the hover labels for this\n            trace.\n        bordercolorsrc\n            Sets the source reference on Chart Studio Cloud for\n            `bordercolor`.\n        font\n            Sets the font used in hover labels.\n        namelength\n            Sets the default length (in number of characters) of\n            the trace name in the hover labels for all traces. -1\n            shows the whole name regardless of length. 0-3 shows\n            the first 0-3 characters, and an integer >3 will show\n            the whole name if it is less than that many characters,\n            but if it is longer, will truncate to `namelength - 3`\n            characters and add an ellipsis.\n        namelengthsrc\n            Sets the source reference on Chart Studio Cloud for\n            `namelength`.\n        '

    def __init__(self, arg=None, align=None, alignsrc=None, bgcolor=None, bgcolorsrc=None, bordercolor=None, bordercolorsrc=None, font=None, namelength=None, namelengthsrc=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Construct a new Hoverlabel object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.pointcloud.Hoverlabel`\n        align\n            Sets the horizontal alignment of the text content\n            within hover label box. Has an effect only if the hover\n            label text spans more two or more lines\n        alignsrc\n            Sets the source reference on Chart Studio Cloud for\n            `align`.\n        bgcolor\n            Sets the background color of the hover labels for this\n            trace\n        bgcolorsrc\n            Sets the source reference on Chart Studio Cloud for\n            `bgcolor`.\n        bordercolor\n            Sets the border color of the hover labels for this\n            trace.\n        bordercolorsrc\n            Sets the source reference on Chart Studio Cloud for\n            `bordercolor`.\n        font\n            Sets the font used in hover labels.\n        namelength\n            Sets the default length (in number of characters) of\n            the trace name in the hover labels for all traces. -1\n            shows the whole name regardless of length. 0-3 shows\n            the first 0-3 characters, and an integer >3 will show\n            the whole name if it is less than that many characters,\n            but if it is longer, will truncate to `namelength - 3`\n            characters and add an ellipsis.\n        namelengthsrc\n            Sets the source reference on Chart Studio Cloud for\n            `namelength`.\n\n        Returns\n        -------\n        Hoverlabel\n        '
        super(Hoverlabel, self).__init__('hoverlabel')
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
            raise ValueError('The first argument to the plotly.graph_objs.pointcloud.Hoverlabel\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.pointcloud.Hoverlabel`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('align', None)
        _v = align if align is not None else _v
        if _v is not None:
            self['align'] = _v
        _v = arg.pop('alignsrc', None)
        _v = alignsrc if alignsrc is not None else _v
        if _v is not None:
            self['alignsrc'] = _v
        _v = arg.pop('bgcolor', None)
        _v = bgcolor if bgcolor is not None else _v
        if _v is not None:
            self['bgcolor'] = _v
        _v = arg.pop('bgcolorsrc', None)
        _v = bgcolorsrc if bgcolorsrc is not None else _v
        if _v is not None:
            self['bgcolorsrc'] = _v
        _v = arg.pop('bordercolor', None)
        _v = bordercolor if bordercolor is not None else _v
        if _v is not None:
            self['bordercolor'] = _v
        _v = arg.pop('bordercolorsrc', None)
        _v = bordercolorsrc if bordercolorsrc is not None else _v
        if _v is not None:
            self['bordercolorsrc'] = _v
        _v = arg.pop('font', None)
        _v = font if font is not None else _v
        if _v is not None:
            self['font'] = _v
        _v = arg.pop('namelength', None)
        _v = namelength if namelength is not None else _v
        if _v is not None:
            self['namelength'] = _v
        _v = arg.pop('namelengthsrc', None)
        _v = namelengthsrc if namelengthsrc is not None else _v
        if _v is not None:
            self['namelengthsrc'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False