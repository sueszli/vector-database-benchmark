from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Updatemenu(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout'
    _path_str = 'layout.updatemenu'
    _valid_props = {'active', 'bgcolor', 'bordercolor', 'borderwidth', 'buttondefaults', 'buttons', 'direction', 'font', 'name', 'pad', 'showactive', 'templateitemname', 'type', 'visible', 'x', 'xanchor', 'y', 'yanchor'}

    @property
    def active(self):
        if False:
            return 10
        "\n        Determines which button (by index starting from 0) is\n        considered active.\n\n        The 'active' property is a integer and may be specified as:\n          - An int (or float that will be cast to an int)\n            in the interval [-1, 9223372036854775807]\n\n        Returns\n        -------\n        int\n        "
        return self['active']

    @active.setter
    def active(self, val):
        if False:
            return 10
        self['active'] = val

    @property
    def bgcolor(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the background color of the update menu buttons.\n\n        The 'bgcolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['bgcolor']

    @bgcolor.setter
    def bgcolor(self, val):
        if False:
            i = 10
            return i + 15
        self['bgcolor'] = val

    @property
    def bordercolor(self):
        if False:
            return 10
        "\n        Sets the color of the border enclosing the update menu.\n\n        The 'bordercolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['bordercolor']

    @bordercolor.setter
    def bordercolor(self, val):
        if False:
            i = 10
            return i + 15
        self['bordercolor'] = val

    @property
    def borderwidth(self):
        if False:
            print('Hello World!')
        "\n        Sets the width (in px) of the border enclosing the update menu.\n\n        The 'borderwidth' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['borderwidth']

    @borderwidth.setter
    def borderwidth(self, val):
        if False:
            return 10
        self['borderwidth'] = val

    @property
    def buttons(self):
        if False:
            i = 10
            return i + 15
        "\n        The 'buttons' property is a tuple of instances of\n        Button that may be specified as:\n          - A list or tuple of instances of plotly.graph_objs.layout.updatemenu.Button\n          - A list or tuple of dicts of string/value properties that\n            will be passed to the Button constructor\n\n            Supported dict properties:\n\n                args\n                    Sets the arguments values to be passed to the\n                    Plotly method set in `method` on click.\n                args2\n                    Sets a 2nd set of `args`, these arguments\n                    values are passed to the Plotly method set in\n                    `method` when clicking this button while in the\n                    active state. Use this to create toggle\n                    buttons.\n                execute\n                    When true, the API method is executed. When\n                    false, all other behaviors are the same and\n                    command execution is skipped. This may be\n                    useful when hooking into, for example, the\n                    `plotly_buttonclicked` method and executing the\n                    API command manually without losing the benefit\n                    of the updatemenu automatically binding to the\n                    state of the plot through the specification of\n                    `method` and `args`.\n                label\n                    Sets the text label to appear on the button.\n                method\n                    Sets the Plotly method to be called on click.\n                    If the `skip` method is used, the API\n                    updatemenu will function as normal but will\n                    perform no API calls and will not bind\n                    automatically to state updates. This may be\n                    used to create a component interface and attach\n                    to updatemenu events manually via JavaScript.\n                name\n                    When used in a template, named items are\n                    created in the output figure in addition to any\n                    items the figure already has in this array. You\n                    can modify these items in the output figure by\n                    making your own item with `templateitemname`\n                    matching this `name` alongside your\n                    modifications (including `visible: false` or\n                    `enabled: false` to hide it). Has no effect\n                    outside of a template.\n                templateitemname\n                    Used to refer to a named item in this array in\n                    the template. Named items from the template\n                    will be created even without a matching item in\n                    the input figure, but you can modify one by\n                    making an item with `templateitemname` matching\n                    its `name`, alongside your modifications\n                    (including `visible: false` or `enabled: false`\n                    to hide it). If there is no template or no\n                    matching item, this item will be hidden unless\n                    you explicitly show it with `visible: true`.\n                visible\n                    Determines whether or not this button is\n                    visible.\n\n        Returns\n        -------\n        tuple[plotly.graph_objs.layout.updatemenu.Button]\n        "
        return self['buttons']

    @buttons.setter
    def buttons(self, val):
        if False:
            print('Hello World!')
        self['buttons'] = val

    @property
    def buttondefaults(self):
        if False:
            while True:
                i = 10
        "\n        When used in a template (as\n        layout.template.layout.updatemenu.buttondefaults), sets the\n        default property values to use for elements of\n        layout.updatemenu.buttons\n\n        The 'buttondefaults' property is an instance of Button\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.updatemenu.Button`\n          - A dict of string/value properties that will be passed\n            to the Button constructor\n\n            Supported dict properties:\n\n        Returns\n        -------\n        plotly.graph_objs.layout.updatemenu.Button\n        "
        return self['buttondefaults']

    @buttondefaults.setter
    def buttondefaults(self, val):
        if False:
            print('Hello World!')
        self['buttondefaults'] = val

    @property
    def direction(self):
        if False:
            return 10
        "\n        Determines the direction in which the buttons are laid out,\n        whether in a dropdown menu or a row/column of buttons. For\n        `left` and `up`, the buttons will still appear in left-to-right\n        or top-to-bottom order respectively.\n\n        The 'direction' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['left', 'right', 'up', 'down']\n\n        Returns\n        -------\n        Any\n        "
        return self['direction']

    @direction.setter
    def direction(self, val):
        if False:
            print('Hello World!')
        self['direction'] = val

    @property
    def font(self):
        if False:
            i = 10
            return i + 15
        '\n        Sets the font of the update menu button text.\n\n        The \'font\' property is an instance of Font\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.updatemenu.Font`\n          - A dict of string/value properties that will be passed\n            to the Font constructor\n\n            Supported dict properties:\n\n                color\n\n                family\n                    HTML font family - the typeface that will be\n                    applied by the web browser. The web browser\n                    will only be able to apply a font if it is\n                    available on the system which it operates.\n                    Provide multiple font families, separated by\n                    commas, to indicate the preference in which to\n                    apply fonts if they aren\'t available on the\n                    system. The Chart Studio Cloud (at\n                    https://chart-studio.plotly.com or on-premise)\n                    generates images on a server, where only a\n                    select number of fonts are installed and\n                    supported. These include "Arial", "Balto",\n                    "Courier New", "Droid Sans",, "Droid Serif",\n                    "Droid Sans Mono", "Gravitas One", "Old\n                    Standard TT", "Open Sans", "Overpass", "PT Sans\n                    Narrow", "Raleway", "Times New Roman".\n                size\n\n        Returns\n        -------\n        plotly.graph_objs.layout.updatemenu.Font\n        '
        return self['font']

    @font.setter
    def font(self, val):
        if False:
            while True:
                i = 10
        self['font'] = val

    @property
    def name(self):
        if False:
            print('Hello World!')
        "\n        When used in a template, named items are created in the output\n        figure in addition to any items the figure already has in this\n        array. You can modify these items in the output figure by\n        making your own item with `templateitemname` matching this\n        `name` alongside your modifications (including `visible: false`\n        or `enabled: false` to hide it). Has no effect outside of a\n        template.\n\n        The 'name' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['name']

    @name.setter
    def name(self, val):
        if False:
            i = 10
            return i + 15
        self['name'] = val

    @property
    def pad(self):
        if False:
            print('Hello World!')
        "\n        Sets the padding around the buttons or dropdown menu.\n\n        The 'pad' property is an instance of Pad\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.updatemenu.Pad`\n          - A dict of string/value properties that will be passed\n            to the Pad constructor\n\n            Supported dict properties:\n\n                b\n                    The amount of padding (in px) along the bottom\n                    of the component.\n                l\n                    The amount of padding (in px) on the left side\n                    of the component.\n                r\n                    The amount of padding (in px) on the right side\n                    of the component.\n                t\n                    The amount of padding (in px) along the top of\n                    the component.\n\n        Returns\n        -------\n        plotly.graph_objs.layout.updatemenu.Pad\n        "
        return self['pad']

    @pad.setter
    def pad(self, val):
        if False:
            return 10
        self['pad'] = val

    @property
    def showactive(self):
        if False:
            while True:
                i = 10
        "\n        Highlights active dropdown item or active button if true.\n\n        The 'showactive' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['showactive']

    @showactive.setter
    def showactive(self, val):
        if False:
            i = 10
            return i + 15
        self['showactive'] = val

    @property
    def templateitemname(self):
        if False:
            while True:
                i = 10
        "\n        Used to refer to a named item in this array in the template.\n        Named items from the template will be created even without a\n        matching item in the input figure, but you can modify one by\n        making an item with `templateitemname` matching its `name`,\n        alongside your modifications (including `visible: false` or\n        `enabled: false` to hide it). If there is no template or no\n        matching item, this item will be hidden unless you explicitly\n        show it with `visible: true`.\n\n        The 'templateitemname' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['templateitemname']

    @templateitemname.setter
    def templateitemname(self, val):
        if False:
            i = 10
            return i + 15
        self['templateitemname'] = val

    @property
    def type(self):
        if False:
            print('Hello World!')
        "\n        Determines whether the buttons are accessible via a dropdown\n        menu or whether the buttons are stacked horizontally or\n        vertically\n\n        The 'type' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['dropdown', 'buttons']\n\n        Returns\n        -------\n        Any\n        "
        return self['type']

    @type.setter
    def type(self, val):
        if False:
            i = 10
            return i + 15
        self['type'] = val

    @property
    def visible(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Determines whether or not the update menu is visible.\n\n        The 'visible' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['visible']

    @visible.setter
    def visible(self, val):
        if False:
            return 10
        self['visible'] = val

    @property
    def x(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the x position (in normalized coordinates) of the update\n        menu.\n\n        The 'x' property is a number and may be specified as:\n          - An int or float in the interval [-2, 3]\n\n        Returns\n        -------\n        int|float\n        "
        return self['x']

    @x.setter
    def x(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['x'] = val

    @property
    def xanchor(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the update menu\'s horizontal position anchor. This anchor\n        binds the `x` position to the "left", "center" or "right" of\n        the range selector.\n\n        The \'xanchor\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'auto\', \'left\', \'center\', \'right\']\n\n        Returns\n        -------\n        Any\n        '
        return self['xanchor']

    @xanchor.setter
    def xanchor(self, val):
        if False:
            i = 10
            return i + 15
        self['xanchor'] = val

    @property
    def y(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the y position (in normalized coordinates) of the update\n        menu.\n\n        The 'y' property is a number and may be specified as:\n          - An int or float in the interval [-2, 3]\n\n        Returns\n        -------\n        int|float\n        "
        return self['y']

    @y.setter
    def y(self, val):
        if False:
            return 10
        self['y'] = val

    @property
    def yanchor(self):
        if False:
            while True:
                i = 10
        '\n        Sets the update menu\'s vertical position anchor This anchor\n        binds the `y` position to the "top", "middle" or "bottom" of\n        the range selector.\n\n        The \'yanchor\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'auto\', \'top\', \'middle\', \'bottom\']\n\n        Returns\n        -------\n        Any\n        '
        return self['yanchor']

    @yanchor.setter
    def yanchor(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['yanchor'] = val

    @property
    def _prop_descriptions(self):
        if False:
            print('Hello World!')
        return '        active\n            Determines which button (by index starting from 0) is\n            considered active.\n        bgcolor\n            Sets the background color of the update menu buttons.\n        bordercolor\n            Sets the color of the border enclosing the update menu.\n        borderwidth\n            Sets the width (in px) of the border enclosing the\n            update menu.\n        buttons\n            A tuple of\n            :class:`plotly.graph_objects.layout.updatemenu.Button`\n            instances or dicts with compatible properties\n        buttondefaults\n            When used in a template (as\n            layout.template.layout.updatemenu.buttondefaults), sets\n            the default property values to use for elements of\n            layout.updatemenu.buttons\n        direction\n            Determines the direction in which the buttons are laid\n            out, whether in a dropdown menu or a row/column of\n            buttons. For `left` and `up`, the buttons will still\n            appear in left-to-right or top-to-bottom order\n            respectively.\n        font\n            Sets the font of the update menu button text.\n        name\n            When used in a template, named items are created in the\n            output figure in addition to any items the figure\n            already has in this array. You can modify these items\n            in the output figure by making your own item with\n            `templateitemname` matching this `name` alongside your\n            modifications (including `visible: false` or `enabled:\n            false` to hide it). Has no effect outside of a\n            template.\n        pad\n            Sets the padding around the buttons or dropdown menu.\n        showactive\n            Highlights active dropdown item or active button if\n            true.\n        templateitemname\n            Used to refer to a named item in this array in the\n            template. Named items from the template will be created\n            even without a matching item in the input figure, but\n            you can modify one by making an item with\n            `templateitemname` matching its `name`, alongside your\n            modifications (including `visible: false` or `enabled:\n            false` to hide it). If there is no template or no\n            matching item, this item will be hidden unless you\n            explicitly show it with `visible: true`.\n        type\n            Determines whether the buttons are accessible via a\n            dropdown menu or whether the buttons are stacked\n            horizontally or vertically\n        visible\n            Determines whether or not the update menu is visible.\n        x\n            Sets the x position (in normalized coordinates) of the\n            update menu.\n        xanchor\n            Sets the update menu\'s horizontal position anchor. This\n            anchor binds the `x` position to the "left", "center"\n            or "right" of the range selector.\n        y\n            Sets the y position (in normalized coordinates) of the\n            update menu.\n        yanchor\n            Sets the update menu\'s vertical position anchor This\n            anchor binds the `y` position to the "top", "middle" or\n            "bottom" of the range selector.\n        '

    def __init__(self, arg=None, active=None, bgcolor=None, bordercolor=None, borderwidth=None, buttons=None, buttondefaults=None, direction=None, font=None, name=None, pad=None, showactive=None, templateitemname=None, type=None, visible=None, x=None, xanchor=None, y=None, yanchor=None, **kwargs):
        if False:
            return 10
        '\n        Construct a new Updatemenu object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.layout.Updatemenu`\n        active\n            Determines which button (by index starting from 0) is\n            considered active.\n        bgcolor\n            Sets the background color of the update menu buttons.\n        bordercolor\n            Sets the color of the border enclosing the update menu.\n        borderwidth\n            Sets the width (in px) of the border enclosing the\n            update menu.\n        buttons\n            A tuple of\n            :class:`plotly.graph_objects.layout.updatemenu.Button`\n            instances or dicts with compatible properties\n        buttondefaults\n            When used in a template (as\n            layout.template.layout.updatemenu.buttondefaults), sets\n            the default property values to use for elements of\n            layout.updatemenu.buttons\n        direction\n            Determines the direction in which the buttons are laid\n            out, whether in a dropdown menu or a row/column of\n            buttons. For `left` and `up`, the buttons will still\n            appear in left-to-right or top-to-bottom order\n            respectively.\n        font\n            Sets the font of the update menu button text.\n        name\n            When used in a template, named items are created in the\n            output figure in addition to any items the figure\n            already has in this array. You can modify these items\n            in the output figure by making your own item with\n            `templateitemname` matching this `name` alongside your\n            modifications (including `visible: false` or `enabled:\n            false` to hide it). Has no effect outside of a\n            template.\n        pad\n            Sets the padding around the buttons or dropdown menu.\n        showactive\n            Highlights active dropdown item or active button if\n            true.\n        templateitemname\n            Used to refer to a named item in this array in the\n            template. Named items from the template will be created\n            even without a matching item in the input figure, but\n            you can modify one by making an item with\n            `templateitemname` matching its `name`, alongside your\n            modifications (including `visible: false` or `enabled:\n            false` to hide it). If there is no template or no\n            matching item, this item will be hidden unless you\n            explicitly show it with `visible: true`.\n        type\n            Determines whether the buttons are accessible via a\n            dropdown menu or whether the buttons are stacked\n            horizontally or vertically\n        visible\n            Determines whether or not the update menu is visible.\n        x\n            Sets the x position (in normalized coordinates) of the\n            update menu.\n        xanchor\n            Sets the update menu\'s horizontal position anchor. This\n            anchor binds the `x` position to the "left", "center"\n            or "right" of the range selector.\n        y\n            Sets the y position (in normalized coordinates) of the\n            update menu.\n        yanchor\n            Sets the update menu\'s vertical position anchor This\n            anchor binds the `y` position to the "top", "middle" or\n            "bottom" of the range selector.\n\n        Returns\n        -------\n        Updatemenu\n        '
        super(Updatemenu, self).__init__('updatemenus')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.Updatemenu\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.Updatemenu`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('active', None)
        _v = active if active is not None else _v
        if _v is not None:
            self['active'] = _v
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
        _v = arg.pop('direction', None)
        _v = direction if direction is not None else _v
        if _v is not None:
            self['direction'] = _v
        _v = arg.pop('font', None)
        _v = font if font is not None else _v
        if _v is not None:
            self['font'] = _v
        _v = arg.pop('name', None)
        _v = name if name is not None else _v
        if _v is not None:
            self['name'] = _v
        _v = arg.pop('pad', None)
        _v = pad if pad is not None else _v
        if _v is not None:
            self['pad'] = _v
        _v = arg.pop('showactive', None)
        _v = showactive if showactive is not None else _v
        if _v is not None:
            self['showactive'] = _v
        _v = arg.pop('templateitemname', None)
        _v = templateitemname if templateitemname is not None else _v
        if _v is not None:
            self['templateitemname'] = _v
        _v = arg.pop('type', None)
        _v = type if type is not None else _v
        if _v is not None:
            self['type'] = _v
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