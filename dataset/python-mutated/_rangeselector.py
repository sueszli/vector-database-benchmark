from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Rangeselector(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout.xaxis'
    _path_str = 'layout.xaxis.rangeselector'
    _valid_props = {'activecolor', 'bgcolor', 'bordercolor', 'borderwidth', 'buttondefaults', 'buttons', 'font', 'visible', 'x', 'xanchor', 'y', 'yanchor'}

    @property
    def activecolor(self):
        if False:
            while True:
                i = 10
        "\n        Sets the background color of the active range selector button.\n\n        The 'activecolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['activecolor']

    @activecolor.setter
    def activecolor(self, val):
        if False:
            i = 10
            return i + 15
        self['activecolor'] = val

    @property
    def bgcolor(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the background color of the range selector buttons.\n\n        The 'bgcolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['bgcolor']

    @bgcolor.setter
    def bgcolor(self, val):
        if False:
            print('Hello World!')
        self['bgcolor'] = val

    @property
    def bordercolor(self):
        if False:
            print('Hello World!')
        "\n        Sets the color of the border enclosing the range selector.\n\n        The 'bordercolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['bordercolor']

    @bordercolor.setter
    def bordercolor(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['bordercolor'] = val

    @property
    def borderwidth(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the width (in px) of the border enclosing the range\n        selector.\n\n        The 'borderwidth' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['borderwidth']

    @borderwidth.setter
    def borderwidth(self, val):
        if False:
            while True:
                i = 10
        self['borderwidth'] = val

    @property
    def buttons(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the specifications for each buttons. By default, a range\n        selector comes with no buttons.\n\n        The \'buttons\' property is a tuple of instances of\n        Button that may be specified as:\n          - A list or tuple of instances of plotly.graph_objs.layout.xaxis.rangeselector.Button\n          - A list or tuple of dicts of string/value properties that\n            will be passed to the Button constructor\n\n            Supported dict properties:\n\n                count\n                    Sets the number of steps to take to update the\n                    range. Use with `step` to specify the update\n                    interval.\n                label\n                    Sets the text label to appear on the button.\n                name\n                    When used in a template, named items are\n                    created in the output figure in addition to any\n                    items the figure already has in this array. You\n                    can modify these items in the output figure by\n                    making your own item with `templateitemname`\n                    matching this `name` alongside your\n                    modifications (including `visible: false` or\n                    `enabled: false` to hide it). Has no effect\n                    outside of a template.\n                step\n                    The unit of measurement that the `count` value\n                    will set the range by.\n                stepmode\n                    Sets the range update mode. If "backward", the\n                    range update shifts the start of range back\n                    "count" times "step" milliseconds. If "todate",\n                    the range update shifts the start of range back\n                    to the first timestamp from "count" times\n                    "step" milliseconds back. For example, with\n                    `step` set to "year" and `count` set to 1 the\n                    range update shifts the start of the range back\n                    to January 01 of the current year. Month and\n                    year "todate" are currently available only for\n                    the built-in (Gregorian) calendar.\n                templateitemname\n                    Used to refer to a named item in this array in\n                    the template. Named items from the template\n                    will be created even without a matching item in\n                    the input figure, but you can modify one by\n                    making an item with `templateitemname` matching\n                    its `name`, alongside your modifications\n                    (including `visible: false` or `enabled: false`\n                    to hide it). If there is no template or no\n                    matching item, this item will be hidden unless\n                    you explicitly show it with `visible: true`.\n                visible\n                    Determines whether or not this button is\n                    visible.\n\n        Returns\n        -------\n        tuple[plotly.graph_objs.layout.xaxis.rangeselector.Button]\n        '
        return self['buttons']

    @buttons.setter
    def buttons(self, val):
        if False:
            return 10
        self['buttons'] = val

    @property
    def buttondefaults(self):
        if False:
            i = 10
            return i + 15
        "\n        When used in a template (as\n        layout.template.layout.xaxis.rangeselector.buttondefaults),\n        sets the default property values to use for elements of\n        layout.xaxis.rangeselector.buttons\n\n        The 'buttondefaults' property is an instance of Button\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.xaxis.rangeselector.Button`\n          - A dict of string/value properties that will be passed\n            to the Button constructor\n\n            Supported dict properties:\n\n        Returns\n        -------\n        plotly.graph_objs.layout.xaxis.rangeselector.Button\n        "
        return self['buttondefaults']

    @buttondefaults.setter
    def buttondefaults(self, val):
        if False:
            i = 10
            return i + 15
        self['buttondefaults'] = val

    @property
    def font(self):
        if False:
            while True:
                i = 10
        '\n        Sets the font of the range selector button text.\n\n        The \'font\' property is an instance of Font\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.xaxis.rangeselector.Font`\n          - A dict of string/value properties that will be passed\n            to the Font constructor\n\n            Supported dict properties:\n\n                color\n\n                family\n                    HTML font family - the typeface that will be\n                    applied by the web browser. The web browser\n                    will only be able to apply a font if it is\n                    available on the system which it operates.\n                    Provide multiple font families, separated by\n                    commas, to indicate the preference in which to\n                    apply fonts if they aren\'t available on the\n                    system. The Chart Studio Cloud (at\n                    https://chart-studio.plotly.com or on-premise)\n                    generates images on a server, where only a\n                    select number of fonts are installed and\n                    supported. These include "Arial", "Balto",\n                    "Courier New", "Droid Sans",, "Droid Serif",\n                    "Droid Sans Mono", "Gravitas One", "Old\n                    Standard TT", "Open Sans", "Overpass", "PT Sans\n                    Narrow", "Raleway", "Times New Roman".\n                size\n\n        Returns\n        -------\n        plotly.graph_objs.layout.xaxis.rangeselector.Font\n        '
        return self['font']

    @font.setter
    def font(self, val):
        if False:
            return 10
        self['font'] = val

    @property
    def visible(self):
        if False:
            return 10
        '\n        Determines whether or not this range selector is visible. Note\n        that range selectors are only available for x axes of `type`\n        set to or auto-typed to "date".\n\n        The \'visible\' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        '
        return self['visible']

    @visible.setter
    def visible(self, val):
        if False:
            i = 10
            return i + 15
        self['visible'] = val

    @property
    def x(self):
        if False:
            print('Hello World!')
        "\n        Sets the x position (in normalized coordinates) of the range\n        selector.\n\n        The 'x' property is a number and may be specified as:\n          - An int or float in the interval [-2, 3]\n\n        Returns\n        -------\n        int|float\n        "
        return self['x']

    @x.setter
    def x(self, val):
        if False:
            return 10
        self['x'] = val

    @property
    def xanchor(self):
        if False:
            print('Hello World!')
        '\n        Sets the range selector\'s horizontal position anchor. This\n        anchor binds the `x` position to the "left", "center" or\n        "right" of the range selector.\n\n        The \'xanchor\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'auto\', \'left\', \'center\', \'right\']\n\n        Returns\n        -------\n        Any\n        '
        return self['xanchor']

    @xanchor.setter
    def xanchor(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['xanchor'] = val

    @property
    def y(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the y position (in normalized coordinates) of the range\n        selector.\n\n        The 'y' property is a number and may be specified as:\n          - An int or float in the interval [-2, 3]\n\n        Returns\n        -------\n        int|float\n        "
        return self['y']

    @y.setter
    def y(self, val):
        if False:
            print('Hello World!')
        self['y'] = val

    @property
    def yanchor(self):
        if False:
            return 10
        '\n        Sets the range selector\'s vertical position anchor This anchor\n        binds the `y` position to the "top", "middle" or "bottom" of\n        the range selector.\n\n        The \'yanchor\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'auto\', \'top\', \'middle\', \'bottom\']\n\n        Returns\n        -------\n        Any\n        '
        return self['yanchor']

    @yanchor.setter
    def yanchor(self, val):
        if False:
            print('Hello World!')
        self['yanchor'] = val

    @property
    def _prop_descriptions(self):
        if False:
            print('Hello World!')
        return '        activecolor\n            Sets the background color of the active range selector\n            button.\n        bgcolor\n            Sets the background color of the range selector\n            buttons.\n        bordercolor\n            Sets the color of the border enclosing the range\n            selector.\n        borderwidth\n            Sets the width (in px) of the border enclosing the\n            range selector.\n        buttons\n            Sets the specifications for each buttons. By default, a\n            range selector comes with no buttons.\n        buttondefaults\n            When used in a template (as layout.template.layout.xaxi\n            s.rangeselector.buttondefaults), sets the default\n            property values to use for elements of\n            layout.xaxis.rangeselector.buttons\n        font\n            Sets the font of the range selector button text.\n        visible\n            Determines whether or not this range selector is\n            visible. Note that range selectors are only available\n            for x axes of `type` set to or auto-typed to "date".\n        x\n            Sets the x position (in normalized coordinates) of the\n            range selector.\n        xanchor\n            Sets the range selector\'s horizontal position anchor.\n            This anchor binds the `x` position to the "left",\n            "center" or "right" of the range selector.\n        y\n            Sets the y position (in normalized coordinates) of the\n            range selector.\n        yanchor\n            Sets the range selector\'s vertical position anchor This\n            anchor binds the `y` position to the "top", "middle" or\n            "bottom" of the range selector.\n        '

    def __init__(self, arg=None, activecolor=None, bgcolor=None, bordercolor=None, borderwidth=None, buttons=None, buttondefaults=None, font=None, visible=None, x=None, xanchor=None, y=None, yanchor=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a new Rangeselector object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.layout.xaxis.Rangeselector`\n        activecolor\n            Sets the background color of the active range selector\n            button.\n        bgcolor\n            Sets the background color of the range selector\n            buttons.\n        bordercolor\n            Sets the color of the border enclosing the range\n            selector.\n        borderwidth\n            Sets the width (in px) of the border enclosing the\n            range selector.\n        buttons\n            Sets the specifications for each buttons. By default, a\n            range selector comes with no buttons.\n        buttondefaults\n            When used in a template (as layout.template.layout.xaxi\n            s.rangeselector.buttondefaults), sets the default\n            property values to use for elements of\n            layout.xaxis.rangeselector.buttons\n        font\n            Sets the font of the range selector button text.\n        visible\n            Determines whether or not this range selector is\n            visible. Note that range selectors are only available\n            for x axes of `type` set to or auto-typed to "date".\n        x\n            Sets the x position (in normalized coordinates) of the\n            range selector.\n        xanchor\n            Sets the range selector\'s horizontal position anchor.\n            This anchor binds the `x` position to the "left",\n            "center" or "right" of the range selector.\n        y\n            Sets the y position (in normalized coordinates) of the\n            range selector.\n        yanchor\n            Sets the range selector\'s vertical position anchor This\n            anchor binds the `y` position to the "top", "middle" or\n            "bottom" of the range selector.\n\n        Returns\n        -------\n        Rangeselector\n        '
        super(Rangeselector, self).__init__('rangeselector')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.xaxis.Rangeselector\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.xaxis.Rangeselector`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('activecolor', None)
        _v = activecolor if activecolor is not None else _v
        if _v is not None:
            self['activecolor'] = _v
        _v = arg.pop('bgcolor', None)
        _v = bgcolor if bgcolor is not None else _v
        if _v is not None:
            self['bgcolor'] = _v
        _v = arg.pop('bordercolor', None)
        _v = bordercolor if bordercolor is not None else _v
        if _v is not None:
            self['bordercolor'] = _v
        _v = arg.pop('borderwidth', None)
        _v = borderwidth if borderwidth is not None else _v
        if _v is not None:
            self['borderwidth'] = _v
        _v = arg.pop('buttons', None)
        _v = buttons if buttons is not None else _v
        if _v is not None:
            self['buttons'] = _v
        _v = arg.pop('buttondefaults', None)
        _v = buttondefaults if buttondefaults is not None else _v
        if _v is not None:
            self['buttondefaults'] = _v
        _v = arg.pop('font', None)
        _v = font if font is not None else _v
        if _v is not None:
            self['font'] = _v
        _v = arg.pop('visible', None)
        _v = visible if visible is not None else _v
        if _v is not None:
            self['visible'] = _v
        _v = arg.pop('x', None)
        _v = x if x is not None else _v
        if _v is not None:
            self['x'] = _v
        _v = arg.pop('xanchor', None)
        _v = xanchor if xanchor is not None else _v
        if _v is not None:
            self['xanchor'] = _v
        _v = arg.pop('y', None)
        _v = y if y is not None else _v
        if _v is not None:
            self['y'] = _v
        _v = arg.pop('yanchor', None)
        _v = yanchor if yanchor is not None else _v
        if _v is not None:
            self['yanchor'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False