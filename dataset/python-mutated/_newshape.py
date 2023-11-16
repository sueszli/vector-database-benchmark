from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Newshape(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout'
    _path_str = 'layout.newshape'
    _valid_props = {'drawdirection', 'fillcolor', 'fillrule', 'label', 'layer', 'legend', 'legendgroup', 'legendgrouptitle', 'legendrank', 'legendwidth', 'line', 'name', 'opacity', 'showlegend', 'visible'}

    @property
    def drawdirection(self):
        if False:
            print('Hello World!')
        '\n        When `dragmode` is set to "drawrect", "drawline" or\n        "drawcircle" this limits the drag to be horizontal, vertical or\n        diagonal. Using "diagonal" there is no limit e.g. in drawing\n        lines in any direction. "ortho" limits the draw to be either\n        horizontal or vertical. "horizontal" allows horizontal extend.\n        "vertical" allows vertical extend.\n\n        The \'drawdirection\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'ortho\', \'horizontal\', \'vertical\', \'diagonal\']\n\n        Returns\n        -------\n        Any\n        '
        return self['drawdirection']

    @drawdirection.setter
    def drawdirection(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['drawdirection'] = val

    @property
    def fillcolor(self):
        if False:
            print('Hello World!')
        "\n        Sets the color filling new shapes' interior. Please note that\n        if using a fillcolor with alpha greater than half, drag inside\n        the active shape starts moving the shape underneath, otherwise\n        a new shape could be started over.\n\n        The 'fillcolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['fillcolor']

    @fillcolor.setter
    def fillcolor(self, val):
        if False:
            while True:
                i = 10
        self['fillcolor'] = val

    @property
    def fillrule(self):
        if False:
            return 10
        "\n        Determines the path's interior. For more info please visit\n        https://developer.mozilla.org/en-\n        US/docs/Web/SVG/Attribute/fill-rule\n\n        The 'fillrule' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['evenodd', 'nonzero']\n\n        Returns\n        -------\n        Any\n        "
        return self['fillrule']

    @fillrule.setter
    def fillrule(self, val):
        if False:
            i = 10
            return i + 15
        self['fillrule'] = val

    @property
    def label(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The \'label\' property is an instance of Label\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.newshape.Label`\n          - A dict of string/value properties that will be passed\n            to the Label constructor\n\n            Supported dict properties:\n\n                font\n                    Sets the new shape label text font.\n                padding\n                    Sets padding (in px) between edge of label and\n                    edge of new shape.\n                text\n                    Sets the text to display with the new shape. It\n                    is also used for legend item if `name` is not\n                    provided.\n                textangle\n                    Sets the angle at which the label text is drawn\n                    with respect to the horizontal. For lines,\n                    angle "auto" is the same angle as the line. For\n                    all other shapes, angle "auto" is horizontal.\n                textposition\n                    Sets the position of the label text relative to\n                    the new shape. Supported values for rectangles,\n                    circles and paths are *top left*, *top center*,\n                    *top right*, *middle left*, *middle center*,\n                    *middle right*, *bottom left*, *bottom center*,\n                    and *bottom right*. Supported values for lines\n                    are "start", "middle", and "end". Default:\n                    *middle center* for rectangles, circles, and\n                    paths; "middle" for lines.\n                texttemplate\n                    Template string used for rendering the new\n                    shape\'s label. Note that this will override\n                    `text`. Variables are inserted using\n                    %{variable}, for example "x0: %{x0}". Numbers\n                    are formatted using d3-format\'s syntax\n                    %{variable:d3-format}, for example "Price:\n                    %{x0:$.2f}". See https://github.com/d3/d3-\n                    format/tree/v1.4.5#d3-format for details on the\n                    formatting syntax. Dates are formatted using\n                    d3-time-format\'s syntax %{variable|d3-time-\n                    format}, for example "Day: %{x0|%m %b %Y}". See\n                    https://github.com/d3/d3-time-\n                    format/tree/v2.2.3#locale_format for details on\n                    the date formatting syntax. A single\n                    multiplication or division operation may be\n                    applied to numeric variables, and combined with\n                    d3 number formatting, for example "Length in\n                    cm: %{x0*2.54}", "%{slope*60:.1f} meters per\n                    second." For log axes, variable values are\n                    given in log units. For date axes, x/y\n                    coordinate variables and center variables use\n                    datetimes, while all other variable values use\n                    values in ms. Finally, the template string has\n                    access to variables `x0`, `x1`, `y0`, `y1`,\n                    `slope`, `dx`, `dy`, `width`, `height`,\n                    `length`, `xcenter` and `ycenter`.\n                xanchor\n                    Sets the label\'s horizontal position anchor\n                    This anchor binds the specified `textposition`\n                    to the "left", "center" or "right" of the label\n                    text. For example, if `textposition` is set to\n                    *top right* and `xanchor` to "right" then the\n                    right-most portion of the label text lines up\n                    with the right-most edge of the new shape.\n                yanchor\n                    Sets the label\'s vertical position anchor This\n                    anchor binds the specified `textposition` to\n                    the "top", "middle" or "bottom" of the label\n                    text. For example, if `textposition` is set to\n                    *top right* and `yanchor` to "top" then the\n                    top-most portion of the label text lines up\n                    with the top-most edge of the new shape.\n\n        Returns\n        -------\n        plotly.graph_objs.layout.newshape.Label\n        '
        return self['label']

    @label.setter
    def label(self, val):
        if False:
            i = 10
            return i + 15
        self['label'] = val

    @property
    def layer(self):
        if False:
            while True:
                i = 10
        "\n        Specifies whether new shapes are drawn below or above traces.\n\n        The 'layer' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['below', 'above']\n\n        Returns\n        -------\n        Any\n        "
        return self['layer']

    @layer.setter
    def layer(self, val):
        if False:
            while True:
                i = 10
        self['layer'] = val

    @property
    def legend(self):
        if False:
            i = 10
            return i + 15
        '\n        Sets the reference to a legend to show new shape in. References\n        to these legends are "legend", "legend2", "legend3", etc.\n        Settings for these legends are set in the layout, under\n        `layout.legend`, `layout.legend2`, etc.\n\n        The \'legend\' property is an identifier of a particular\n        subplot, of type \'legend\', that may be specified as the string \'legend\'\n        optionally followed by an integer >= 1\n        (e.g. \'legend\', \'legend1\', \'legend2\', \'legend3\', etc.)\n\n        Returns\n        -------\n        str\n        '
        return self['legend']

    @legend.setter
    def legend(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['legend'] = val

    @property
    def legendgroup(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the legend group for new shape. Traces and shapes part of\n        the same legend group hide/show at the same time when toggling\n        legend items.\n\n        The 'legendgroup' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['legendgroup']

    @legendgroup.setter
    def legendgroup(self, val):
        if False:
            while True:
                i = 10
        self['legendgroup'] = val

    @property
    def legendgrouptitle(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The 'legendgrouptitle' property is an instance of Legendgrouptitle\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.newshape.Legendgrouptitle`\n          - A dict of string/value properties that will be passed\n            to the Legendgrouptitle constructor\n\n            Supported dict properties:\n\n                font\n                    Sets this legend group's title font.\n                text\n                    Sets the title of the legend group.\n\n        Returns\n        -------\n        plotly.graph_objs.layout.newshape.Legendgrouptitle\n        "
        return self['legendgrouptitle']

    @legendgrouptitle.setter
    def legendgrouptitle(self, val):
        if False:
            print('Hello World!')
        self['legendgrouptitle'] = val

    @property
    def legendrank(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the legend rank for new shape. Items and groups with\n        smaller ranks are presented on top/left side while with\n        "reversed" `legend.traceorder` they are on bottom/right side.\n        The default legendrank is 1000, so that you can use ranks less\n        than 1000 to place certain items before all unranked items, and\n        ranks greater than 1000 to go after all unranked items.\n\n        The \'legendrank\' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        '
        return self['legendrank']

    @legendrank.setter
    def legendrank(self, val):
        if False:
            print('Hello World!')
        self['legendrank'] = val

    @property
    def legendwidth(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the width (in px or fraction) of the legend for new shape.\n\n        The 'legendwidth' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['legendwidth']

    @legendwidth.setter
    def legendwidth(self, val):
        if False:
            return 10
        self['legendwidth'] = val

    @property
    def line(self):
        if False:
            while True:
                i = 10
        '\n        The \'line\' property is an instance of Line\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.newshape.Line`\n          - A dict of string/value properties that will be passed\n            to the Line constructor\n\n            Supported dict properties:\n\n                color\n                    Sets the line color. By default uses either\n                    dark grey or white to increase contrast with\n                    background color.\n                dash\n                    Sets the dash style of lines. Set to a dash\n                    type string ("solid", "dot", "dash",\n                    "longdash", "dashdot", or "longdashdot") or a\n                    dash length list in px (eg "5px,10px,2px,2px").\n                width\n                    Sets the line width (in px).\n\n        Returns\n        -------\n        plotly.graph_objs.layout.newshape.Line\n        '
        return self['line']

    @line.setter
    def line(self, val):
        if False:
            print('Hello World!')
        self['line'] = val

    @property
    def name(self):
        if False:
            return 10
        "\n        Sets new shape name. The name appears as the legend item.\n\n        The 'name' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['name']

    @name.setter
    def name(self, val):
        if False:
            while True:
                i = 10
        self['name'] = val

    @property
    def opacity(self):
        if False:
            return 10
        "\n        Sets the opacity of new shapes.\n\n        The 'opacity' property is a number and may be specified as:\n          - An int or float in the interval [0, 1]\n\n        Returns\n        -------\n        int|float\n        "
        return self['opacity']

    @opacity.setter
    def opacity(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['opacity'] = val

    @property
    def showlegend(self):
        if False:
            while True:
                i = 10
        "\n        Determines whether or not new shape is shown in the legend.\n\n        The 'showlegend' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['showlegend']

    @showlegend.setter
    def showlegend(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['showlegend'] = val

    @property
    def visible(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Determines whether or not new shape is visible. If\n        "legendonly", the shape is not drawn, but can appear as a\n        legend item (provided that the legend itself is visible).\n\n        The \'visible\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [True, False, \'legendonly\']\n\n        Returns\n        -------\n        Any\n        '
        return self['visible']

    @visible.setter
    def visible(self, val):
        if False:
            while True:
                i = 10
        self['visible'] = val

    @property
    def _prop_descriptions(self):
        if False:
            print('Hello World!')
        return '        drawdirection\n            When `dragmode` is set to "drawrect", "drawline" or\n            "drawcircle" this limits the drag to be horizontal,\n            vertical or diagonal. Using "diagonal" there is no\n            limit e.g. in drawing lines in any direction. "ortho"\n            limits the draw to be either horizontal or vertical.\n            "horizontal" allows horizontal extend. "vertical"\n            allows vertical extend.\n        fillcolor\n            Sets the color filling new shapes\' interior. Please\n            note that if using a fillcolor with alpha greater than\n            half, drag inside the active shape starts moving the\n            shape underneath, otherwise a new shape could be\n            started over.\n        fillrule\n            Determines the path\'s interior. For more info please\n            visit https://developer.mozilla.org/en-\n            US/docs/Web/SVG/Attribute/fill-rule\n        label\n            :class:`plotly.graph_objects.layout.newshape.Label`\n            instance or dict with compatible properties\n        layer\n            Specifies whether new shapes are drawn below or above\n            traces.\n        legend\n            Sets the reference to a legend to show new shape in.\n            References to these legends are "legend", "legend2",\n            "legend3", etc. Settings for these legends are set in\n            the layout, under `layout.legend`, `layout.legend2`,\n            etc.\n        legendgroup\n            Sets the legend group for new shape. Traces and shapes\n            part of the same legend group hide/show at the same\n            time when toggling legend items.\n        legendgrouptitle\n            :class:`plotly.graph_objects.layout.newshape.Legendgrou\n            ptitle` instance or dict with compatible properties\n        legendrank\n            Sets the legend rank for new shape. Items and groups\n            with smaller ranks are presented on top/left side while\n            with "reversed" `legend.traceorder` they are on\n            bottom/right side. The default legendrank is 1000, so\n            that you can use ranks less than 1000 to place certain\n            items before all unranked items, and ranks greater than\n            1000 to go after all unranked items.\n        legendwidth\n            Sets the width (in px or fraction) of the legend for\n            new shape.\n        line\n            :class:`plotly.graph_objects.layout.newshape.Line`\n            instance or dict with compatible properties\n        name\n            Sets new shape name. The name appears as the legend\n            item.\n        opacity\n            Sets the opacity of new shapes.\n        showlegend\n            Determines whether or not new shape is shown in the\n            legend.\n        visible\n            Determines whether or not new shape is visible. If\n            "legendonly", the shape is not drawn, but can appear as\n            a legend item (provided that the legend itself is\n            visible).\n        '

    def __init__(self, arg=None, drawdirection=None, fillcolor=None, fillrule=None, label=None, layer=None, legend=None, legendgroup=None, legendgrouptitle=None, legendrank=None, legendwidth=None, line=None, name=None, opacity=None, showlegend=None, visible=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Construct a new Newshape object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.layout.Newshape`\n        drawdirection\n            When `dragmode` is set to "drawrect", "drawline" or\n            "drawcircle" this limits the drag to be horizontal,\n            vertical or diagonal. Using "diagonal" there is no\n            limit e.g. in drawing lines in any direction. "ortho"\n            limits the draw to be either horizontal or vertical.\n            "horizontal" allows horizontal extend. "vertical"\n            allows vertical extend.\n        fillcolor\n            Sets the color filling new shapes\' interior. Please\n            note that if using a fillcolor with alpha greater than\n            half, drag inside the active shape starts moving the\n            shape underneath, otherwise a new shape could be\n            started over.\n        fillrule\n            Determines the path\'s interior. For more info please\n            visit https://developer.mozilla.org/en-\n            US/docs/Web/SVG/Attribute/fill-rule\n        label\n            :class:`plotly.graph_objects.layout.newshape.Label`\n            instance or dict with compatible properties\n        layer\n            Specifies whether new shapes are drawn below or above\n            traces.\n        legend\n            Sets the reference to a legend to show new shape in.\n            References to these legends are "legend", "legend2",\n            "legend3", etc. Settings for these legends are set in\n            the layout, under `layout.legend`, `layout.legend2`,\n            etc.\n        legendgroup\n            Sets the legend group for new shape. Traces and shapes\n            part of the same legend group hide/show at the same\n            time when toggling legend items.\n        legendgrouptitle\n            :class:`plotly.graph_objects.layout.newshape.Legendgrou\n            ptitle` instance or dict with compatible properties\n        legendrank\n            Sets the legend rank for new shape. Items and groups\n            with smaller ranks are presented on top/left side while\n            with "reversed" `legend.traceorder` they are on\n            bottom/right side. The default legendrank is 1000, so\n            that you can use ranks less than 1000 to place certain\n            items before all unranked items, and ranks greater than\n            1000 to go after all unranked items.\n        legendwidth\n            Sets the width (in px or fraction) of the legend for\n            new shape.\n        line\n            :class:`plotly.graph_objects.layout.newshape.Line`\n            instance or dict with compatible properties\n        name\n            Sets new shape name. The name appears as the legend\n            item.\n        opacity\n            Sets the opacity of new shapes.\n        showlegend\n            Determines whether or not new shape is shown in the\n            legend.\n        visible\n            Determines whether or not new shape is visible. If\n            "legendonly", the shape is not drawn, but can appear as\n            a legend item (provided that the legend itself is\n            visible).\n\n        Returns\n        -------\n        Newshape\n        '
        super(Newshape, self).__init__('newshape')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.Newshape\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.Newshape`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('drawdirection', None)
        _v = drawdirection if drawdirection is not None else _v
        if _v is not None:
            self['drawdirection'] = _v
        _v = arg.pop('fillcolor', None)
        _v = fillcolor if fillcolor is not None else _v
        if _v is not None:
            self['fillcolor'] = _v
        _v = arg.pop('fillrule', None)
        _v = fillrule if fillrule is not None else _v
        if _v is not None:
            self['fillrule'] = _v
        _v = arg.pop('label', None)
        _v = label if label is not None else _v
        if _v is not None:
            self['label'] = _v
        _v = arg.pop('layer', None)
        _v = layer if layer is not None else _v
        if _v is not None:
            self['layer'] = _v
        _v = arg.pop('legend', None)
        _v = legend if legend is not None else _v
        if _v is not None:
            self['legend'] = _v
        _v = arg.pop('legendgroup', None)
        _v = legendgroup if legendgroup is not None else _v
        if _v is not None:
            self['legendgroup'] = _v
        _v = arg.pop('legendgrouptitle', None)
        _v = legendgrouptitle if legendgrouptitle is not None else _v
        if _v is not None:
            self['legendgrouptitle'] = _v
        _v = arg.pop('legendrank', None)
        _v = legendrank if legendrank is not None else _v
        if _v is not None:
            self['legendrank'] = _v
        _v = arg.pop('legendwidth', None)
        _v = legendwidth if legendwidth is not None else _v
        if _v is not None:
            self['legendwidth'] = _v
        _v = arg.pop('line', None)
        _v = line if line is not None else _v
        if _v is not None:
            self['line'] = _v
        _v = arg.pop('name', None)
        _v = name if name is not None else _v
        if _v is not None:
            self['name'] = _v
        _v = arg.pop('opacity', None)
        _v = opacity if opacity is not None else _v
        if _v is not None:
            self['opacity'] = _v
        _v = arg.pop('showlegend', None)
        _v = showlegend if showlegend is not None else _v
        if _v is not None:
            self['showlegend'] = _v
        _v = arg.pop('visible', None)
        _v = visible if visible is not None else _v
        if _v is not None:
            self['visible'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False