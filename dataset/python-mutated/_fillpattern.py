from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class Fillpattern(_BaseTraceHierarchyType):
    _parent_path_str = 'scatter'
    _path_str = 'scatter.fillpattern'
    _valid_props = {'bgcolor', 'bgcolorsrc', 'fgcolor', 'fgcolorsrc', 'fgopacity', 'fillmode', 'shape', 'shapesrc', 'size', 'sizesrc', 'solidity', 'soliditysrc'}

    @property
    def bgcolor(self):
        if False:
            return 10
        '\n        When there is no colorscale sets the color of background\n        pattern fill. Defaults to a `marker.color` background when\n        `fillmode` is "overlay". Otherwise, defaults to a transparent\n        background.\n\n        The \'bgcolor\' property is a color and may be specified as:\n          - A hex string (e.g. \'#ff0000\')\n          - An rgb/rgba string (e.g. \'rgb(255,0,0)\')\n          - An hsl/hsla string (e.g. \'hsl(0,100%,50%)\')\n          - An hsv/hsva string (e.g. \'hsv(0,100%,100%)\')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n          - A list or array of any of the above\n\n        Returns\n        -------\n        str|numpy.ndarray\n        '
        return self['bgcolor']

    @bgcolor.setter
    def bgcolor(self, val):
        if False:
            i = 10
            return i + 15
        self['bgcolor'] = val

    @property
    def bgcolorsrc(self):
        if False:
            print('Hello World!')
        "\n        Sets the source reference on Chart Studio Cloud for `bgcolor`.\n\n        The 'bgcolorsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['bgcolorsrc']

    @bgcolorsrc.setter
    def bgcolorsrc(self, val):
        if False:
            i = 10
            return i + 15
        self['bgcolorsrc'] = val

    @property
    def fgcolor(self):
        if False:
            print('Hello World!')
        '\n        When there is no colorscale sets the color of foreground\n        pattern fill. Defaults to a `marker.color` background when\n        `fillmode` is "replace". Otherwise, defaults to dark grey or\n        white to increase contrast with the `bgcolor`.\n\n        The \'fgcolor\' property is a color and may be specified as:\n          - A hex string (e.g. \'#ff0000\')\n          - An rgb/rgba string (e.g. \'rgb(255,0,0)\')\n          - An hsl/hsla string (e.g. \'hsl(0,100%,50%)\')\n          - An hsv/hsva string (e.g. \'hsv(0,100%,100%)\')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n          - A list or array of any of the above\n\n        Returns\n        -------\n        str|numpy.ndarray\n        '
        return self['fgcolor']

    @fgcolor.setter
    def fgcolor(self, val):
        if False:
            i = 10
            return i + 15
        self['fgcolor'] = val

    @property
    def fgcolorsrc(self):
        if False:
            while True:
                i = 10
        "\n        Sets the source reference on Chart Studio Cloud for `fgcolor`.\n\n        The 'fgcolorsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['fgcolorsrc']

    @fgcolorsrc.setter
    def fgcolorsrc(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['fgcolorsrc'] = val

    @property
    def fgopacity(self):
        if False:
            i = 10
            return i + 15
        '\n        Sets the opacity of the foreground pattern fill. Defaults to a\n        0.5 when `fillmode` is "overlay". Otherwise, defaults to 1.\n\n        The \'fgopacity\' property is a number and may be specified as:\n          - An int or float in the interval [0, 1]\n\n        Returns\n        -------\n        int|float\n        '
        return self['fgopacity']

    @fgopacity.setter
    def fgopacity(self, val):
        if False:
            print('Hello World!')
        self['fgopacity'] = val

    @property
    def fillmode(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Determines whether `marker.color` should be used as a default\n        to `bgcolor` or a `fgcolor`.\n\n        The 'fillmode' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['replace', 'overlay']\n\n        Returns\n        -------\n        Any\n        "
        return self['fillmode']

    @fillmode.setter
    def fillmode(self, val):
        if False:
            return 10
        self['fillmode'] = val

    @property
    def shape(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the shape of the pattern fill. By default, no pattern is\n        used for filling the area.\n\n        The 'shape' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['', '/', '\\', 'x', '-', '|', '+', '.']\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        Any|numpy.ndarray\n        "
        return self['shape']

    @shape.setter
    def shape(self, val):
        if False:
            i = 10
            return i + 15
        self['shape'] = val

    @property
    def shapesrc(self):
        if False:
            while True:
                i = 10
        "\n        Sets the source reference on Chart Studio Cloud for `shape`.\n\n        The 'shapesrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['shapesrc']

    @shapesrc.setter
    def shapesrc(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['shapesrc'] = val

    @property
    def size(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the size of unit squares of the pattern fill in pixels,\n        which corresponds to the interval of repetition of the pattern.\n\n        The 'size' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        int|float|numpy.ndarray\n        "
        return self['size']

    @size.setter
    def size(self, val):
        if False:
            i = 10
            return i + 15
        self['size'] = val

    @property
    def sizesrc(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the source reference on Chart Studio Cloud for `size`.\n\n        The 'sizesrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['sizesrc']

    @sizesrc.setter
    def sizesrc(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['sizesrc'] = val

    @property
    def solidity(self):
        if False:
            print('Hello World!')
        "\n        Sets the solidity of the pattern fill. Solidity is roughly the\n        fraction of the area filled by the pattern. Solidity of 0 shows\n        only the background color without pattern and solidty of 1\n        shows only the foreground color without pattern.\n\n        The 'solidity' property is a number and may be specified as:\n          - An int or float in the interval [0, 1]\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        int|float|numpy.ndarray\n        "
        return self['solidity']

    @solidity.setter
    def solidity(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['solidity'] = val

    @property
    def soliditysrc(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the source reference on Chart Studio Cloud for `solidity`.\n\n        The 'soliditysrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['soliditysrc']

    @soliditysrc.setter
    def soliditysrc(self, val):
        if False:
            i = 10
            return i + 15
        self['soliditysrc'] = val

    @property
    def _prop_descriptions(self):
        if False:
            print('Hello World!')
        return '        bgcolor\n            When there is no colorscale sets the color of\n            background pattern fill. Defaults to a `marker.color`\n            background when `fillmode` is "overlay". Otherwise,\n            defaults to a transparent background.\n        bgcolorsrc\n            Sets the source reference on Chart Studio Cloud for\n            `bgcolor`.\n        fgcolor\n            When there is no colorscale sets the color of\n            foreground pattern fill. Defaults to a `marker.color`\n            background when `fillmode` is "replace". Otherwise,\n            defaults to dark grey or white to increase contrast\n            with the `bgcolor`.\n        fgcolorsrc\n            Sets the source reference on Chart Studio Cloud for\n            `fgcolor`.\n        fgopacity\n            Sets the opacity of the foreground pattern fill.\n            Defaults to a 0.5 when `fillmode` is "overlay".\n            Otherwise, defaults to 1.\n        fillmode\n            Determines whether `marker.color` should be used as a\n            default to `bgcolor` or a `fgcolor`.\n        shape\n            Sets the shape of the pattern fill. By default, no\n            pattern is used for filling the area.\n        shapesrc\n            Sets the source reference on Chart Studio Cloud for\n            `shape`.\n        size\n            Sets the size of unit squares of the pattern fill in\n            pixels, which corresponds to the interval of repetition\n            of the pattern.\n        sizesrc\n            Sets the source reference on Chart Studio Cloud for\n            `size`.\n        solidity\n            Sets the solidity of the pattern fill. Solidity is\n            roughly the fraction of the area filled by the pattern.\n            Solidity of 0 shows only the background color without\n            pattern and solidty of 1 shows only the foreground\n            color without pattern.\n        soliditysrc\n            Sets the source reference on Chart Studio Cloud for\n            `solidity`.\n        '

    def __init__(self, arg=None, bgcolor=None, bgcolorsrc=None, fgcolor=None, fgcolorsrc=None, fgopacity=None, fillmode=None, shape=None, shapesrc=None, size=None, sizesrc=None, solidity=None, soliditysrc=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Construct a new Fillpattern object\n\n        Sets the pattern within the marker.\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.scatter.Fillpattern`\n        bgcolor\n            When there is no colorscale sets the color of\n            background pattern fill. Defaults to a `marker.color`\n            background when `fillmode` is "overlay". Otherwise,\n            defaults to a transparent background.\n        bgcolorsrc\n            Sets the source reference on Chart Studio Cloud for\n            `bgcolor`.\n        fgcolor\n            When there is no colorscale sets the color of\n            foreground pattern fill. Defaults to a `marker.color`\n            background when `fillmode` is "replace". Otherwise,\n            defaults to dark grey or white to increase contrast\n            with the `bgcolor`.\n        fgcolorsrc\n            Sets the source reference on Chart Studio Cloud for\n            `fgcolor`.\n        fgopacity\n            Sets the opacity of the foreground pattern fill.\n            Defaults to a 0.5 when `fillmode` is "overlay".\n            Otherwise, defaults to 1.\n        fillmode\n            Determines whether `marker.color` should be used as a\n            default to `bgcolor` or a `fgcolor`.\n        shape\n            Sets the shape of the pattern fill. By default, no\n            pattern is used for filling the area.\n        shapesrc\n            Sets the source reference on Chart Studio Cloud for\n            `shape`.\n        size\n            Sets the size of unit squares of the pattern fill in\n            pixels, which corresponds to the interval of repetition\n            of the pattern.\n        sizesrc\n            Sets the source reference on Chart Studio Cloud for\n            `size`.\n        solidity\n            Sets the solidity of the pattern fill. Solidity is\n            roughly the fraction of the area filled by the pattern.\n            Solidity of 0 shows only the background color without\n            pattern and solidty of 1 shows only the foreground\n            color without pattern.\n        soliditysrc\n            Sets the source reference on Chart Studio Cloud for\n            `solidity`.\n\n        Returns\n        -------\n        Fillpattern\n        '
        super(Fillpattern, self).__init__('fillpattern')
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
            raise ValueError('The first argument to the plotly.graph_objs.scatter.Fillpattern\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.scatter.Fillpattern`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('bgcolor', None)
        _v = bgcolor if bgcolor is not None else _v
        if _v is not None:
            self['bgcolor'] = _v
        _v = arg.pop('bgcolorsrc', None)
        _v = bgcolorsrc if bgcolorsrc is not None else _v
        if _v is not None:
            self['bgcolorsrc'] = _v
        _v = arg.pop('fgcolor', None)
        _v = fgcolor if fgcolor is not None else _v
        if _v is not None:
            self['fgcolor'] = _v
        _v = arg.pop('fgcolorsrc', None)
        _v = fgcolorsrc if fgcolorsrc is not None else _v
        if _v is not None:
            self['fgcolorsrc'] = _v
        _v = arg.pop('fgopacity', None)
        _v = fgopacity if fgopacity is not None else _v
        if _v is not None:
            self['fgopacity'] = _v
        _v = arg.pop('fillmode', None)
        _v = fillmode if fillmode is not None else _v
        if _v is not None:
            self['fillmode'] = _v
        _v = arg.pop('shape', None)
        _v = shape if shape is not None else _v
        if _v is not None:
            self['shape'] = _v
        _v = arg.pop('shapesrc', None)
        _v = shapesrc if shapesrc is not None else _v
        if _v is not None:
            self['shapesrc'] = _v
        _v = arg.pop('size', None)
        _v = size if size is not None else _v
        if _v is not None:
            self['size'] = _v
        _v = arg.pop('sizesrc', None)
        _v = sizesrc if sizesrc is not None else _v
        if _v is not None:
            self['sizesrc'] = _v
        _v = arg.pop('solidity', None)
        _v = solidity if solidity is not None else _v
        if _v is not None:
            self['solidity'] = _v
        _v = arg.pop('soliditysrc', None)
        _v = soliditysrc if soliditysrc is not None else _v
        if _v is not None:
            self['soliditysrc'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False