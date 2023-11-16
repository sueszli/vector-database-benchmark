from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Annotation(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout'
    _path_str = 'layout.annotation'
    _valid_props = {'align', 'arrowcolor', 'arrowhead', 'arrowside', 'arrowsize', 'arrowwidth', 'ax', 'axref', 'ay', 'ayref', 'bgcolor', 'bordercolor', 'borderpad', 'borderwidth', 'captureevents', 'clicktoshow', 'font', 'height', 'hoverlabel', 'hovertext', 'name', 'opacity', 'showarrow', 'standoff', 'startarrowhead', 'startarrowsize', 'startstandoff', 'templateitemname', 'text', 'textangle', 'valign', 'visible', 'width', 'x', 'xanchor', 'xclick', 'xref', 'xshift', 'y', 'yanchor', 'yclick', 'yref', 'yshift'}

    @property
    def align(self):
        if False:
            while True:
                i = 10
        "\n        Sets the horizontal alignment of the `text` within the box. Has\n        an effect only if `text` spans two or more lines (i.e. `text`\n        contains one or more <br> HTML tags) or if an explicit width is\n        set to override the text width.\n\n        The 'align' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['left', 'center', 'right']\n\n        Returns\n        -------\n        Any\n        "
        return self['align']

    @align.setter
    def align(self, val):
        if False:
            print('Hello World!')
        self['align'] = val

    @property
    def arrowcolor(self):
        if False:
            return 10
        "\n        Sets the color of the annotation arrow.\n\n        The 'arrowcolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['arrowcolor']

    @arrowcolor.setter
    def arrowcolor(self, val):
        if False:
            i = 10
            return i + 15
        self['arrowcolor'] = val

    @property
    def arrowhead(self):
        if False:
            print('Hello World!')
        "\n        Sets the end annotation arrow head style.\n\n        The 'arrowhead' property is a integer and may be specified as:\n          - An int (or float that will be cast to an int)\n            in the interval [0, 8]\n\n        Returns\n        -------\n        int\n        "
        return self['arrowhead']

    @arrowhead.setter
    def arrowhead(self, val):
        if False:
            i = 10
            return i + 15
        self['arrowhead'] = val

    @property
    def arrowside(self):
        if False:
            return 10
        "\n        Sets the annotation arrow head position.\n\n        The 'arrowside' property is a flaglist and may be specified\n        as a string containing:\n          - Any combination of ['end', 'start'] joined with '+' characters\n            (e.g. 'end+start')\n            OR exactly one of ['none'] (e.g. 'none')\n\n        Returns\n        -------\n        Any\n        "
        return self['arrowside']

    @arrowside.setter
    def arrowside(self, val):
        if False:
            return 10
        self['arrowside'] = val

    @property
    def arrowsize(self):
        if False:
            return 10
        "\n        Sets the size of the end annotation arrow head, relative to\n        `arrowwidth`. A value of 1 (default) gives a head about 3x as\n        wide as the line.\n\n        The 'arrowsize' property is a number and may be specified as:\n          - An int or float in the interval [0.3, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['arrowsize']

    @arrowsize.setter
    def arrowsize(self, val):
        if False:
            return 10
        self['arrowsize'] = val

    @property
    def arrowwidth(self):
        if False:
            print('Hello World!')
        "\n        Sets the width (in px) of annotation arrow line.\n\n        The 'arrowwidth' property is a number and may be specified as:\n          - An int or float in the interval [0.1, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['arrowwidth']

    @arrowwidth.setter
    def arrowwidth(self, val):
        if False:
            while True:
                i = 10
        self['arrowwidth'] = val

    @property
    def ax(self):
        if False:
            print('Hello World!')
        "\n        Sets the x component of the arrow tail about the arrow head. If\n        `axref` is `pixel`, a positive (negative) component corresponds\n        to an arrow pointing from right to left (left to right). If\n        `axref` is not `pixel` and is exactly the same as `xref`, this\n        is an absolute value on that axis, like `x`, specified in the\n        same coordinates as `xref`.\n\n        The 'ax' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['ax']

    @ax.setter
    def ax(self, val):
        if False:
            while True:
                i = 10
        self['ax'] = val

    @property
    def axref(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Indicates in what coordinates the tail of the annotation\n        (ax,ay) is specified. If set to a x axis id (e.g. "x" or "x2"),\n        the `x` position refers to a x coordinate. If set to "paper",\n        the `x` position refers to the distance from the left of the\n        plotting area in normalized coordinates where 0 (1) corresponds\n        to the left (right). If set to a x axis ID followed by "domain"\n        (separated by a space), the position behaves like for "paper",\n        but refers to the distance in fractions of the domain length\n        from the left of the domain of that axis: e.g., *x2 domain*\n        refers to the domain of the second x  axis and a x position of\n        0.5 refers to the point between the left and the right of the\n        domain of the second x axis. In order for absolute positioning\n        of the arrow to work, "axref" must be exactly the same as\n        "xref", otherwise "axref" will revert to "pixel" (explained\n        next). For relative positioning, "axref" can be set to "pixel",\n        in which case the "ax" value is specified in pixels relative to\n        "x". Absolute positioning is useful for trendline annotations\n        which should continue to indicate the correct trend when\n        zoomed. Relative positioning is useful for specifying the text\n        offset for an annotated point.\n\n        The \'axref\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'pixel\']\n          - A string that matches one of the following regular expressions:\n                [\'^x([2-9]|[1-9][0-9]+)?( domain)?$\']\n\n        Returns\n        -------\n        Any\n        '
        return self['axref']

    @axref.setter
    def axref(self, val):
        if False:
            return 10
        self['axref'] = val

    @property
    def ay(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the y component of the arrow tail about the arrow head. If\n        `ayref` is `pixel`, a positive (negative) component corresponds\n        to an arrow pointing from bottom to top (top to bottom). If\n        `ayref` is not `pixel` and is exactly the same as `yref`, this\n        is an absolute value on that axis, like `y`, specified in the\n        same coordinates as `yref`.\n\n        The 'ay' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['ay']

    @ay.setter
    def ay(self, val):
        if False:
            return 10
        self['ay'] = val

    @property
    def ayref(self):
        if False:
            i = 10
            return i + 15
        '\n        Indicates in what coordinates the tail of the annotation\n        (ax,ay) is specified. If set to a y axis id (e.g. "y" or "y2"),\n        the `y` position refers to a y coordinate. If set to "paper",\n        the `y` position refers to the distance from the bottom of the\n        plotting area in normalized coordinates where 0 (1) corresponds\n        to the bottom (top). If set to a y axis ID followed by "domain"\n        (separated by a space), the position behaves like for "paper",\n        but refers to the distance in fractions of the domain length\n        from the bottom of the domain of that axis: e.g., *y2 domain*\n        refers to the domain of the second y  axis and a y position of\n        0.5 refers to the point between the bottom and the top of the\n        domain of the second y axis. In order for absolute positioning\n        of the arrow to work, "ayref" must be exactly the same as\n        "yref", otherwise "ayref" will revert to "pixel" (explained\n        next). For relative positioning, "ayref" can be set to "pixel",\n        in which case the "ay" value is specified in pixels relative to\n        "y". Absolute positioning is useful for trendline annotations\n        which should continue to indicate the correct trend when\n        zoomed. Relative positioning is useful for specifying the text\n        offset for an annotated point.\n\n        The \'ayref\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'pixel\']\n          - A string that matches one of the following regular expressions:\n                [\'^y([2-9]|[1-9][0-9]+)?( domain)?$\']\n\n        Returns\n        -------\n        Any\n        '
        return self['ayref']

    @ayref.setter
    def ayref(self, val):
        if False:
            i = 10
            return i + 15
        self['ayref'] = val

    @property
    def bgcolor(self):
        if False:
            while True:
                i = 10
        "\n        Sets the background color of the annotation.\n\n        The 'bgcolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
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
            i = 10
            return i + 15
        "\n        Sets the color of the border enclosing the annotation `text`.\n\n        The 'bordercolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['bordercolor']

    @bordercolor.setter
    def bordercolor(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['bordercolor'] = val

    @property
    def borderpad(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the padding (in px) between the `text` and the enclosing\n        border.\n\n        The 'borderpad' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['borderpad']

    @borderpad.setter
    def borderpad(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['borderpad'] = val

    @property
    def borderwidth(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the width (in px) of the border enclosing the annotation\n        `text`.\n\n        The 'borderwidth' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['borderwidth']

    @borderwidth.setter
    def borderwidth(self, val):
        if False:
            while True:
                i = 10
        self['borderwidth'] = val

    @property
    def captureevents(self):
        if False:
            return 10
        "\n        Determines whether the annotation text box captures mouse move\n        and click events, or allows those events to pass through to\n        data points in the plot that may be behind the annotation. By\n        default `captureevents` is False unless `hovertext` is\n        provided. If you use the event `plotly_clickannotation` without\n        `hovertext` you must explicitly enable `captureevents`.\n\n        The 'captureevents' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['captureevents']

    @captureevents.setter
    def captureevents(self, val):
        if False:
            print('Hello World!')
        self['captureevents'] = val

    @property
    def clicktoshow(self):
        if False:
            i = 10
            return i + 15
        '\n        Makes this annotation respond to clicks on the plot. If you\n        click a data point that exactly matches the `x` and `y` values\n        of this annotation, and it is hidden (visible: false), it will\n        appear. In "onoff" mode, you must click the same point again to\n        make it disappear, so if you click multiple points, you can\n        show multiple annotations. In "onout" mode, a click anywhere\n        else in the plot (on another data point or not) will hide this\n        annotation. If you need to show/hide this annotation in\n        response to different `x` or `y` values, you can set `xclick`\n        and/or `yclick`. This is useful for example to label the side\n        of a bar. To label markers though, `standoff` is preferred over\n        `xclick` and `yclick`.\n\n        The \'clicktoshow\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [False, \'onoff\', \'onout\']\n\n        Returns\n        -------\n        Any\n        '
        return self['clicktoshow']

    @clicktoshow.setter
    def clicktoshow(self, val):
        if False:
            print('Hello World!')
        self['clicktoshow'] = val

    @property
    def font(self):
        if False:
            return 10
        '\n        Sets the annotation text font.\n\n        The \'font\' property is an instance of Font\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.annotation.Font`\n          - A dict of string/value properties that will be passed\n            to the Font constructor\n\n            Supported dict properties:\n\n                color\n\n                family\n                    HTML font family - the typeface that will be\n                    applied by the web browser. The web browser\n                    will only be able to apply a font if it is\n                    available on the system which it operates.\n                    Provide multiple font families, separated by\n                    commas, to indicate the preference in which to\n                    apply fonts if they aren\'t available on the\n                    system. The Chart Studio Cloud (at\n                    https://chart-studio.plotly.com or on-premise)\n                    generates images on a server, where only a\n                    select number of fonts are installed and\n                    supported. These include "Arial", "Balto",\n                    "Courier New", "Droid Sans",, "Droid Serif",\n                    "Droid Sans Mono", "Gravitas One", "Old\n                    Standard TT", "Open Sans", "Overpass", "PT Sans\n                    Narrow", "Raleway", "Times New Roman".\n                size\n\n        Returns\n        -------\n        plotly.graph_objs.layout.annotation.Font\n        '
        return self['font']

    @font.setter
    def font(self, val):
        if False:
            while True:
                i = 10
        self['font'] = val

    @property
    def height(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets an explicit height for the text box. null (default) lets\n        the text set the box height. Taller text will be clipped.\n\n        The 'height' property is a number and may be specified as:\n          - An int or float in the interval [1, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['height']

    @height.setter
    def height(self, val):
        if False:
            while True:
                i = 10
        self['height'] = val

    @property
    def hoverlabel(self):
        if False:
            return 10
        "\n        The 'hoverlabel' property is an instance of Hoverlabel\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.annotation.Hoverlabel`\n          - A dict of string/value properties that will be passed\n            to the Hoverlabel constructor\n\n            Supported dict properties:\n\n                bgcolor\n                    Sets the background color of the hover label.\n                    By default uses the annotation's `bgcolor` made\n                    opaque, or white if it was transparent.\n                bordercolor\n                    Sets the border color of the hover label. By\n                    default uses either dark grey or white, for\n                    maximum contrast with `hoverlabel.bgcolor`.\n                font\n                    Sets the hover label text font. By default uses\n                    the global hover font and size, with color from\n                    `hoverlabel.bordercolor`.\n\n        Returns\n        -------\n        plotly.graph_objs.layout.annotation.Hoverlabel\n        "
        return self['hoverlabel']

    @hoverlabel.setter
    def hoverlabel(self, val):
        if False:
            while True:
                i = 10
        self['hoverlabel'] = val

    @property
    def hovertext(self):
        if False:
            while True:
                i = 10
        "\n        Sets text to appear when hovering over this annotation. If\n        omitted or blank, no hover label will appear.\n\n        The 'hovertext' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['hovertext']

    @hovertext.setter
    def hovertext(self, val):
        if False:
            return 10
        self['hovertext'] = val

    @property
    def name(self):
        if False:
            print('Hello World!')
        "\n        When used in a template, named items are created in the output\n        figure in addition to any items the figure already has in this\n        array. You can modify these items in the output figure by\n        making your own item with `templateitemname` matching this\n        `name` alongside your modifications (including `visible: false`\n        or `enabled: false` to hide it). Has no effect outside of a\n        template.\n\n        The 'name' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
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
            while True:
                i = 10
        "\n        Sets the opacity of the annotation (text + arrow).\n\n        The 'opacity' property is a number and may be specified as:\n          - An int or float in the interval [0, 1]\n\n        Returns\n        -------\n        int|float\n        "
        return self['opacity']

    @opacity.setter
    def opacity(self, val):
        if False:
            return 10
        self['opacity'] = val

    @property
    def showarrow(self):
        if False:
            i = 10
            return i + 15
        "\n        Determines whether or not the annotation is drawn with an\n        arrow. If True, `text` is placed near the arrow's tail. If\n        False, `text` lines up with the `x` and `y` provided.\n\n        The 'showarrow' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['showarrow']

    @showarrow.setter
    def showarrow(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['showarrow'] = val

    @property
    def standoff(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets a distance, in pixels, to move the end arrowhead away from\n        the position it is pointing at, for example to point at the\n        edge of a marker independent of zoom. Note that this shortens\n        the arrow from the `ax` / `ay` vector, in contrast to `xshift`\n        / `yshift` which moves everything by this amount.\n\n        The 'standoff' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['standoff']

    @standoff.setter
    def standoff(self, val):
        if False:
            print('Hello World!')
        self['standoff'] = val

    @property
    def startarrowhead(self):
        if False:
            while True:
                i = 10
        "\n        Sets the start annotation arrow head style.\n\n        The 'startarrowhead' property is a integer and may be specified as:\n          - An int (or float that will be cast to an int)\n            in the interval [0, 8]\n\n        Returns\n        -------\n        int\n        "
        return self['startarrowhead']

    @startarrowhead.setter
    def startarrowhead(self, val):
        if False:
            return 10
        self['startarrowhead'] = val

    @property
    def startarrowsize(self):
        if False:
            while True:
                i = 10
        "\n        Sets the size of the start annotation arrow head, relative to\n        `arrowwidth`. A value of 1 (default) gives a head about 3x as\n        wide as the line.\n\n        The 'startarrowsize' property is a number and may be specified as:\n          - An int or float in the interval [0.3, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['startarrowsize']

    @startarrowsize.setter
    def startarrowsize(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['startarrowsize'] = val

    @property
    def startstandoff(self):
        if False:
            while True:
                i = 10
        "\n        Sets a distance, in pixels, to move the start arrowhead away\n        from the position it is pointing at, for example to point at\n        the edge of a marker independent of zoom. Note that this\n        shortens the arrow from the `ax` / `ay` vector, in contrast to\n        `xshift` / `yshift` which moves everything by this amount.\n\n        The 'startstandoff' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['startstandoff']

    @startstandoff.setter
    def startstandoff(self, val):
        if False:
            return 10
        self['startstandoff'] = val

    @property
    def templateitemname(self):
        if False:
            print('Hello World!')
        "\n        Used to refer to a named item in this array in the template.\n        Named items from the template will be created even without a\n        matching item in the input figure, but you can modify one by\n        making an item with `templateitemname` matching its `name`,\n        alongside your modifications (including `visible: false` or\n        `enabled: false` to hide it). If there is no template or no\n        matching item, this item will be hidden unless you explicitly\n        show it with `visible: true`.\n\n        The 'templateitemname' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['templateitemname']

    @templateitemname.setter
    def templateitemname(self, val):
        if False:
            i = 10
            return i + 15
        self['templateitemname'] = val

    @property
    def text(self):
        if False:
            print('Hello World!')
        "\n        Sets the text associated with this annotation. Plotly uses a\n        subset of HTML tags to do things like newline (<br>), bold\n        (<b></b>), italics (<i></i>), hyperlinks (<a href='...'></a>).\n        Tags <em>, <sup>, <sub> <span> are also supported.\n\n        The 'text' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['text']

    @text.setter
    def text(self, val):
        if False:
            print('Hello World!')
        self['text'] = val

    @property
    def textangle(self):
        if False:
            while True:
                i = 10
        "\n        Sets the angle at which the `text` is drawn with respect to the\n        horizontal.\n\n        The 'textangle' property is a angle (in degrees) that may be\n        specified as a number between -180 and 180.\n        Numeric values outside this range are converted to the equivalent value\n        (e.g. 270 is converted to -90).\n\n        Returns\n        -------\n        int|float\n        "
        return self['textangle']

    @textangle.setter
    def textangle(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['textangle'] = val

    @property
    def valign(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the vertical alignment of the `text` within the box. Has\n        an effect only if an explicit height is set to override the\n        text height.\n\n        The 'valign' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['top', 'middle', 'bottom']\n\n        Returns\n        -------\n        Any\n        "
        return self['valign']

    @valign.setter
    def valign(self, val):
        if False:
            i = 10
            return i + 15
        self['valign'] = val

    @property
    def visible(self):
        if False:
            while True:
                i = 10
        "\n        Determines whether or not this annotation is visible.\n\n        The 'visible' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['visible']

    @visible.setter
    def visible(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['visible'] = val

    @property
    def width(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets an explicit width for the text box. null (default) lets\n        the text set the box width. Wider text will be clipped. There\n        is no automatic wrapping; use <br> to start a new line.\n\n        The 'width' property is a number and may be specified as:\n          - An int or float in the interval [1, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['width']

    @width.setter
    def width(self, val):
        if False:
            print('Hello World!')
        self['width'] = val

    @property
    def x(self):
        if False:
            print('Hello World!')
        '\n        Sets the annotation\'s x position. If the axis `type` is "log",\n        then you must take the log of your desired range. If the axis\n        `type` is "date", it should be date strings, like date data,\n        though Date objects and unix milliseconds will be accepted and\n        converted to strings. If the axis `type` is "category", it\n        should be numbers, using the scale where each category is\n        assigned a serial number from zero in the order it appears.\n\n        The \'x\' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        '
        return self['x']

    @x.setter
    def x(self, val):
        if False:
            print('Hello World!')
        self['x'] = val

    @property
    def xanchor(self):
        if False:
            i = 10
            return i + 15
        '\n        Sets the text box\'s horizontal position anchor This anchor\n        binds the `x` position to the "left", "center" or "right" of\n        the annotation. For example, if `x` is set to 1, `xref` to\n        "paper" and `xanchor` to "right" then the right-most portion of\n        the annotation lines up with the right-most edge of the\n        plotting area. If "auto", the anchor is equivalent to "center"\n        for data-referenced annotations or if there is an arrow,\n        whereas for paper-referenced with no arrow, the anchor picked\n        corresponds to the closest side.\n\n        The \'xanchor\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'auto\', \'left\', \'center\', \'right\']\n\n        Returns\n        -------\n        Any\n        '
        return self['xanchor']

    @xanchor.setter
    def xanchor(self, val):
        if False:
            print('Hello World!')
        self['xanchor'] = val

    @property
    def xclick(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Toggle this annotation when clicking a data point whose `x`\n        value is `xclick` rather than the annotation's `x` value.\n\n        The 'xclick' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['xclick']

    @xclick.setter
    def xclick(self, val):
        if False:
            while True:
                i = 10
        self['xclick'] = val

    @property
    def xref(self):
        if False:
            i = 10
            return i + 15
        '\n        Sets the annotation\'s x coordinate axis. If set to a x axis id\n        (e.g. "x" or "x2"), the `x` position refers to a x coordinate.\n        If set to "paper", the `x` position refers to the distance from\n        the left of the plotting area in normalized coordinates where 0\n        (1) corresponds to the left (right). If set to a x axis ID\n        followed by "domain" (separated by a space), the position\n        behaves like for "paper", but refers to the distance in\n        fractions of the domain length from the left of the domain of\n        that axis: e.g., *x2 domain* refers to the domain of the second\n        x  axis and a x position of 0.5 refers to the point between the\n        left and the right of the domain of the second x axis.\n\n        The \'xref\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'paper\']\n          - A string that matches one of the following regular expressions:\n                [\'^x([2-9]|[1-9][0-9]+)?( domain)?$\']\n\n        Returns\n        -------\n        Any\n        '
        return self['xref']

    @xref.setter
    def xref(self, val):
        if False:
            while True:
                i = 10
        self['xref'] = val

    @property
    def xshift(self):
        if False:
            while True:
                i = 10
        "\n        Shifts the position of the whole annotation and arrow to the\n        right (positive) or left (negative) by this many pixels.\n\n        The 'xshift' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['xshift']

    @xshift.setter
    def xshift(self, val):
        if False:
            return 10
        self['xshift'] = val

    @property
    def y(self):
        if False:
            while True:
                i = 10
        '\n        Sets the annotation\'s y position. If the axis `type` is "log",\n        then you must take the log of your desired range. If the axis\n        `type` is "date", it should be date strings, like date data,\n        though Date objects and unix milliseconds will be accepted and\n        converted to strings. If the axis `type` is "category", it\n        should be numbers, using the scale where each category is\n        assigned a serial number from zero in the order it appears.\n\n        The \'y\' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        '
        return self['y']

    @y.setter
    def y(self, val):
        if False:
            return 10
        self['y'] = val

    @property
    def yanchor(self):
        if False:
            return 10
        '\n        Sets the text box\'s vertical position anchor This anchor binds\n        the `y` position to the "top", "middle" or "bottom" of the\n        annotation. For example, if `y` is set to 1, `yref` to "paper"\n        and `yanchor` to "top" then the top-most portion of the\n        annotation lines up with the top-most edge of the plotting\n        area. If "auto", the anchor is equivalent to "middle" for data-\n        referenced annotations or if there is an arrow, whereas for\n        paper-referenced with no arrow, the anchor picked corresponds\n        to the closest side.\n\n        The \'yanchor\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'auto\', \'top\', \'middle\', \'bottom\']\n\n        Returns\n        -------\n        Any\n        '
        return self['yanchor']

    @yanchor.setter
    def yanchor(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['yanchor'] = val

    @property
    def yclick(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Toggle this annotation when clicking a data point whose `y`\n        value is `yclick` rather than the annotation's `y` value.\n\n        The 'yclick' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['yclick']

    @yclick.setter
    def yclick(self, val):
        if False:
            i = 10
            return i + 15
        self['yclick'] = val

    @property
    def yref(self):
        if False:
            return 10
        '\n        Sets the annotation\'s y coordinate axis. If set to a y axis id\n        (e.g. "y" or "y2"), the `y` position refers to a y coordinate.\n        If set to "paper", the `y` position refers to the distance from\n        the bottom of the plotting area in normalized coordinates where\n        0 (1) corresponds to the bottom (top). If set to a y axis ID\n        followed by "domain" (separated by a space), the position\n        behaves like for "paper", but refers to the distance in\n        fractions of the domain length from the bottom of the domain of\n        that axis: e.g., *y2 domain* refers to the domain of the second\n        y  axis and a y position of 0.5 refers to the point between the\n        bottom and the top of the domain of the second y axis.\n\n        The \'yref\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'paper\']\n          - A string that matches one of the following regular expressions:\n                [\'^y([2-9]|[1-9][0-9]+)?( domain)?$\']\n\n        Returns\n        -------\n        Any\n        '
        return self['yref']

    @yref.setter
    def yref(self, val):
        if False:
            while True:
                i = 10
        self['yref'] = val

    @property
    def yshift(self):
        if False:
            while True:
                i = 10
        "\n        Shifts the position of the whole annotation and arrow up\n        (positive) or down (negative) by this many pixels.\n\n        The 'yshift' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['yshift']

    @yshift.setter
    def yshift(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['yshift'] = val

    @property
    def _prop_descriptions(self):
        if False:
            for i in range(10):
                print('nop')
        return '        align\n            Sets the horizontal alignment of the `text` within the\n            box. Has an effect only if `text` spans two or more\n            lines (i.e. `text` contains one or more <br> HTML tags)\n            or if an explicit width is set to override the text\n            width.\n        arrowcolor\n            Sets the color of the annotation arrow.\n        arrowhead\n            Sets the end annotation arrow head style.\n        arrowside\n            Sets the annotation arrow head position.\n        arrowsize\n            Sets the size of the end annotation arrow head,\n            relative to `arrowwidth`. A value of 1 (default) gives\n            a head about 3x as wide as the line.\n        arrowwidth\n            Sets the width (in px) of annotation arrow line.\n        ax\n            Sets the x component of the arrow tail about the arrow\n            head. If `axref` is `pixel`, a positive (negative)\n            component corresponds to an arrow pointing from right\n            to left (left to right). If `axref` is not `pixel` and\n            is exactly the same as `xref`, this is an absolute\n            value on that axis, like `x`, specified in the same\n            coordinates as `xref`.\n        axref\n            Indicates in what coordinates the tail of the\n            annotation (ax,ay) is specified. If set to a x axis id\n            (e.g. "x" or "x2"), the `x` position refers to a x\n            coordinate. If set to "paper", the `x` position refers\n            to the distance from the left of the plotting area in\n            normalized coordinates where 0 (1) corresponds to the\n            left (right). If set to a x axis ID followed by\n            "domain" (separated by a space), the position behaves\n            like for "paper", but refers to the distance in\n            fractions of the domain length from the left of the\n            domain of that axis: e.g., *x2 domain* refers to the\n            domain of the second x  axis and a x position of 0.5\n            refers to the point between the left and the right of\n            the domain of the second x axis. In order for absolute\n            positioning of the arrow to work, "axref" must be\n            exactly the same as "xref", otherwise "axref" will\n            revert to "pixel" (explained next). For relative\n            positioning, "axref" can be set to "pixel", in which\n            case the "ax" value is specified in pixels relative to\n            "x". Absolute positioning is useful for trendline\n            annotations which should continue to indicate the\n            correct trend when zoomed. Relative positioning is\n            useful for specifying the text offset for an annotated\n            point.\n        ay\n            Sets the y component of the arrow tail about the arrow\n            head. If `ayref` is `pixel`, a positive (negative)\n            component corresponds to an arrow pointing from bottom\n            to top (top to bottom). If `ayref` is not `pixel` and\n            is exactly the same as `yref`, this is an absolute\n            value on that axis, like `y`, specified in the same\n            coordinates as `yref`.\n        ayref\n            Indicates in what coordinates the tail of the\n            annotation (ax,ay) is specified. If set to a y axis id\n            (e.g. "y" or "y2"), the `y` position refers to a y\n            coordinate. If set to "paper", the `y` position refers\n            to the distance from the bottom of the plotting area in\n            normalized coordinates where 0 (1) corresponds to the\n            bottom (top). If set to a y axis ID followed by\n            "domain" (separated by a space), the position behaves\n            like for "paper", but refers to the distance in\n            fractions of the domain length from the bottom of the\n            domain of that axis: e.g., *y2 domain* refers to the\n            domain of the second y  axis and a y position of 0.5\n            refers to the point between the bottom and the top of\n            the domain of the second y axis. In order for absolute\n            positioning of the arrow to work, "ayref" must be\n            exactly the same as "yref", otherwise "ayref" will\n            revert to "pixel" (explained next). For relative\n            positioning, "ayref" can be set to "pixel", in which\n            case the "ay" value is specified in pixels relative to\n            "y". Absolute positioning is useful for trendline\n            annotations which should continue to indicate the\n            correct trend when zoomed. Relative positioning is\n            useful for specifying the text offset for an annotated\n            point.\n        bgcolor\n            Sets the background color of the annotation.\n        bordercolor\n            Sets the color of the border enclosing the annotation\n            `text`.\n        borderpad\n            Sets the padding (in px) between the `text` and the\n            enclosing border.\n        borderwidth\n            Sets the width (in px) of the border enclosing the\n            annotation `text`.\n        captureevents\n            Determines whether the annotation text box captures\n            mouse move and click events, or allows those events to\n            pass through to data points in the plot that may be\n            behind the annotation. By default `captureevents` is\n            False unless `hovertext` is provided. If you use the\n            event `plotly_clickannotation` without `hovertext` you\n            must explicitly enable `captureevents`.\n        clicktoshow\n            Makes this annotation respond to clicks on the plot. If\n            you click a data point that exactly matches the `x` and\n            `y` values of this annotation, and it is hidden\n            (visible: false), it will appear. In "onoff" mode, you\n            must click the same point again to make it disappear,\n            so if you click multiple points, you can show multiple\n            annotations. In "onout" mode, a click anywhere else in\n            the plot (on another data point or not) will hide this\n            annotation. If you need to show/hide this annotation in\n            response to different `x` or `y` values, you can set\n            `xclick` and/or `yclick`. This is useful for example to\n            label the side of a bar. To label markers though,\n            `standoff` is preferred over `xclick` and `yclick`.\n        font\n            Sets the annotation text font.\n        height\n            Sets an explicit height for the text box. null\n            (default) lets the text set the box height. Taller text\n            will be clipped.\n        hoverlabel\n            :class:`plotly.graph_objects.layout.annotation.Hoverlab\n            el` instance or dict with compatible properties\n        hovertext\n            Sets text to appear when hovering over this annotation.\n            If omitted or blank, no hover label will appear.\n        name\n            When used in a template, named items are created in the\n            output figure in addition to any items the figure\n            already has in this array. You can modify these items\n            in the output figure by making your own item with\n            `templateitemname` matching this `name` alongside your\n            modifications (including `visible: false` or `enabled:\n            false` to hide it). Has no effect outside of a\n            template.\n        opacity\n            Sets the opacity of the annotation (text + arrow).\n        showarrow\n            Determines whether or not the annotation is drawn with\n            an arrow. If True, `text` is placed near the arrow\'s\n            tail. If False, `text` lines up with the `x` and `y`\n            provided.\n        standoff\n            Sets a distance, in pixels, to move the end arrowhead\n            away from the position it is pointing at, for example\n            to point at the edge of a marker independent of zoom.\n            Note that this shortens the arrow from the `ax` / `ay`\n            vector, in contrast to `xshift` / `yshift` which moves\n            everything by this amount.\n        startarrowhead\n            Sets the start annotation arrow head style.\n        startarrowsize\n            Sets the size of the start annotation arrow head,\n            relative to `arrowwidth`. A value of 1 (default) gives\n            a head about 3x as wide as the line.\n        startstandoff\n            Sets a distance, in pixels, to move the start arrowhead\n            away from the position it is pointing at, for example\n            to point at the edge of a marker independent of zoom.\n            Note that this shortens the arrow from the `ax` / `ay`\n            vector, in contrast to `xshift` / `yshift` which moves\n            everything by this amount.\n        templateitemname\n            Used to refer to a named item in this array in the\n            template. Named items from the template will be created\n            even without a matching item in the input figure, but\n            you can modify one by making an item with\n            `templateitemname` matching its `name`, alongside your\n            modifications (including `visible: false` or `enabled:\n            false` to hide it). If there is no template or no\n            matching item, this item will be hidden unless you\n            explicitly show it with `visible: true`.\n        text\n            Sets the text associated with this annotation. Plotly\n            uses a subset of HTML tags to do things like newline\n            (<br>), bold (<b></b>), italics (<i></i>), hyperlinks\n            (<a href=\'...\'></a>). Tags <em>, <sup>, <sub> <span>\n            are also supported.\n        textangle\n            Sets the angle at which the `text` is drawn with\n            respect to the horizontal.\n        valign\n            Sets the vertical alignment of the `text` within the\n            box. Has an effect only if an explicit height is set to\n            override the text height.\n        visible\n            Determines whether or not this annotation is visible.\n        width\n            Sets an explicit width for the text box. null (default)\n            lets the text set the box width. Wider text will be\n            clipped. There is no automatic wrapping; use <br> to\n            start a new line.\n        x\n            Sets the annotation\'s x position. If the axis `type` is\n            "log", then you must take the log of your desired\n            range. If the axis `type` is "date", it should be date\n            strings, like date data, though Date objects and unix\n            milliseconds will be accepted and converted to strings.\n            If the axis `type` is "category", it should be numbers,\n            using the scale where each category is assigned a\n            serial number from zero in the order it appears.\n        xanchor\n            Sets the text box\'s horizontal position anchor This\n            anchor binds the `x` position to the "left", "center"\n            or "right" of the annotation. For example, if `x` is\n            set to 1, `xref` to "paper" and `xanchor` to "right"\n            then the right-most portion of the annotation lines up\n            with the right-most edge of the plotting area. If\n            "auto", the anchor is equivalent to "center" for data-\n            referenced annotations or if there is an arrow, whereas\n            for paper-referenced with no arrow, the anchor picked\n            corresponds to the closest side.\n        xclick\n            Toggle this annotation when clicking a data point whose\n            `x` value is `xclick` rather than the annotation\'s `x`\n            value.\n        xref\n            Sets the annotation\'s x coordinate axis. If set to a x\n            axis id (e.g. "x" or "x2"), the `x` position refers to\n            a x coordinate. If set to "paper", the `x` position\n            refers to the distance from the left of the plotting\n            area in normalized coordinates where 0 (1) corresponds\n            to the left (right). If set to a x axis ID followed by\n            "domain" (separated by a space), the position behaves\n            like for "paper", but refers to the distance in\n            fractions of the domain length from the left of the\n            domain of that axis: e.g., *x2 domain* refers to the\n            domain of the second x  axis and a x position of 0.5\n            refers to the point between the left and the right of\n            the domain of the second x axis.\n        xshift\n            Shifts the position of the whole annotation and arrow\n            to the right (positive) or left (negative) by this many\n            pixels.\n        y\n            Sets the annotation\'s y position. If the axis `type` is\n            "log", then you must take the log of your desired\n            range. If the axis `type` is "date", it should be date\n            strings, like date data, though Date objects and unix\n            milliseconds will be accepted and converted to strings.\n            If the axis `type` is "category", it should be numbers,\n            using the scale where each category is assigned a\n            serial number from zero in the order it appears.\n        yanchor\n            Sets the text box\'s vertical position anchor This\n            anchor binds the `y` position to the "top", "middle" or\n            "bottom" of the annotation. For example, if `y` is set\n            to 1, `yref` to "paper" and `yanchor` to "top" then the\n            top-most portion of the annotation lines up with the\n            top-most edge of the plotting area. If "auto", the\n            anchor is equivalent to "middle" for data-referenced\n            annotations or if there is an arrow, whereas for paper-\n            referenced with no arrow, the anchor picked corresponds\n            to the closest side.\n        yclick\n            Toggle this annotation when clicking a data point whose\n            `y` value is `yclick` rather than the annotation\'s `y`\n            value.\n        yref\n            Sets the annotation\'s y coordinate axis. If set to a y\n            axis id (e.g. "y" or "y2"), the `y` position refers to\n            a y coordinate. If set to "paper", the `y` position\n            refers to the distance from the bottom of the plotting\n            area in normalized coordinates where 0 (1) corresponds\n            to the bottom (top). If set to a y axis ID followed by\n            "domain" (separated by a space), the position behaves\n            like for "paper", but refers to the distance in\n            fractions of the domain length from the bottom of the\n            domain of that axis: e.g., *y2 domain* refers to the\n            domain of the second y  axis and a y position of 0.5\n            refers to the point between the bottom and the top of\n            the domain of the second y axis.\n        yshift\n            Shifts the position of the whole annotation and arrow\n            up (positive) or down (negative) by this many pixels.\n        '

    def __init__(self, arg=None, align=None, arrowcolor=None, arrowhead=None, arrowside=None, arrowsize=None, arrowwidth=None, ax=None, axref=None, ay=None, ayref=None, bgcolor=None, bordercolor=None, borderpad=None, borderwidth=None, captureevents=None, clicktoshow=None, font=None, height=None, hoverlabel=None, hovertext=None, name=None, opacity=None, showarrow=None, standoff=None, startarrowhead=None, startarrowsize=None, startstandoff=None, templateitemname=None, text=None, textangle=None, valign=None, visible=None, width=None, x=None, xanchor=None, xclick=None, xref=None, xshift=None, y=None, yanchor=None, yclick=None, yref=None, yshift=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a new Annotation object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.layout.Annotation`\n        align\n            Sets the horizontal alignment of the `text` within the\n            box. Has an effect only if `text` spans two or more\n            lines (i.e. `text` contains one or more <br> HTML tags)\n            or if an explicit width is set to override the text\n            width.\n        arrowcolor\n            Sets the color of the annotation arrow.\n        arrowhead\n            Sets the end annotation arrow head style.\n        arrowside\n            Sets the annotation arrow head position.\n        arrowsize\n            Sets the size of the end annotation arrow head,\n            relative to `arrowwidth`. A value of 1 (default) gives\n            a head about 3x as wide as the line.\n        arrowwidth\n            Sets the width (in px) of annotation arrow line.\n        ax\n            Sets the x component of the arrow tail about the arrow\n            head. If `axref` is `pixel`, a positive (negative)\n            component corresponds to an arrow pointing from right\n            to left (left to right). If `axref` is not `pixel` and\n            is exactly the same as `xref`, this is an absolute\n            value on that axis, like `x`, specified in the same\n            coordinates as `xref`.\n        axref\n            Indicates in what coordinates the tail of the\n            annotation (ax,ay) is specified. If set to a x axis id\n            (e.g. "x" or "x2"), the `x` position refers to a x\n            coordinate. If set to "paper", the `x` position refers\n            to the distance from the left of the plotting area in\n            normalized coordinates where 0 (1) corresponds to the\n            left (right). If set to a x axis ID followed by\n            "domain" (separated by a space), the position behaves\n            like for "paper", but refers to the distance in\n            fractions of the domain length from the left of the\n            domain of that axis: e.g., *x2 domain* refers to the\n            domain of the second x  axis and a x position of 0.5\n            refers to the point between the left and the right of\n            the domain of the second x axis. In order for absolute\n            positioning of the arrow to work, "axref" must be\n            exactly the same as "xref", otherwise "axref" will\n            revert to "pixel" (explained next). For relative\n            positioning, "axref" can be set to "pixel", in which\n            case the "ax" value is specified in pixels relative to\n            "x". Absolute positioning is useful for trendline\n            annotations which should continue to indicate the\n            correct trend when zoomed. Relative positioning is\n            useful for specifying the text offset for an annotated\n            point.\n        ay\n            Sets the y component of the arrow tail about the arrow\n            head. If `ayref` is `pixel`, a positive (negative)\n            component corresponds to an arrow pointing from bottom\n            to top (top to bottom). If `ayref` is not `pixel` and\n            is exactly the same as `yref`, this is an absolute\n            value on that axis, like `y`, specified in the same\n            coordinates as `yref`.\n        ayref\n            Indicates in what coordinates the tail of the\n            annotation (ax,ay) is specified. If set to a y axis id\n            (e.g. "y" or "y2"), the `y` position refers to a y\n            coordinate. If set to "paper", the `y` position refers\n            to the distance from the bottom of the plotting area in\n            normalized coordinates where 0 (1) corresponds to the\n            bottom (top). If set to a y axis ID followed by\n            "domain" (separated by a space), the position behaves\n            like for "paper", but refers to the distance in\n            fractions of the domain length from the bottom of the\n            domain of that axis: e.g., *y2 domain* refers to the\n            domain of the second y  axis and a y position of 0.5\n            refers to the point between the bottom and the top of\n            the domain of the second y axis. In order for absolute\n            positioning of the arrow to work, "ayref" must be\n            exactly the same as "yref", otherwise "ayref" will\n            revert to "pixel" (explained next). For relative\n            positioning, "ayref" can be set to "pixel", in which\n            case the "ay" value is specified in pixels relative to\n            "y". Absolute positioning is useful for trendline\n            annotations which should continue to indicate the\n            correct trend when zoomed. Relative positioning is\n            useful for specifying the text offset for an annotated\n            point.\n        bgcolor\n            Sets the background color of the annotation.\n        bordercolor\n            Sets the color of the border enclosing the annotation\n            `text`.\n        borderpad\n            Sets the padding (in px) between the `text` and the\n            enclosing border.\n        borderwidth\n            Sets the width (in px) of the border enclosing the\n            annotation `text`.\n        captureevents\n            Determines whether the annotation text box captures\n            mouse move and click events, or allows those events to\n            pass through to data points in the plot that may be\n            behind the annotation. By default `captureevents` is\n            False unless `hovertext` is provided. If you use the\n            event `plotly_clickannotation` without `hovertext` you\n            must explicitly enable `captureevents`.\n        clicktoshow\n            Makes this annotation respond to clicks on the plot. If\n            you click a data point that exactly matches the `x` and\n            `y` values of this annotation, and it is hidden\n            (visible: false), it will appear. In "onoff" mode, you\n            must click the same point again to make it disappear,\n            so if you click multiple points, you can show multiple\n            annotations. In "onout" mode, a click anywhere else in\n            the plot (on another data point or not) will hide this\n            annotation. If you need to show/hide this annotation in\n            response to different `x` or `y` values, you can set\n            `xclick` and/or `yclick`. This is useful for example to\n            label the side of a bar. To label markers though,\n            `standoff` is preferred over `xclick` and `yclick`.\n        font\n            Sets the annotation text font.\n        height\n            Sets an explicit height for the text box. null\n            (default) lets the text set the box height. Taller text\n            will be clipped.\n        hoverlabel\n            :class:`plotly.graph_objects.layout.annotation.Hoverlab\n            el` instance or dict with compatible properties\n        hovertext\n            Sets text to appear when hovering over this annotation.\n            If omitted or blank, no hover label will appear.\n        name\n            When used in a template, named items are created in the\n            output figure in addition to any items the figure\n            already has in this array. You can modify these items\n            in the output figure by making your own item with\n            `templateitemname` matching this `name` alongside your\n            modifications (including `visible: false` or `enabled:\n            false` to hide it). Has no effect outside of a\n            template.\n        opacity\n            Sets the opacity of the annotation (text + arrow).\n        showarrow\n            Determines whether or not the annotation is drawn with\n            an arrow. If True, `text` is placed near the arrow\'s\n            tail. If False, `text` lines up with the `x` and `y`\n            provided.\n        standoff\n            Sets a distance, in pixels, to move the end arrowhead\n            away from the position it is pointing at, for example\n            to point at the edge of a marker independent of zoom.\n            Note that this shortens the arrow from the `ax` / `ay`\n            vector, in contrast to `xshift` / `yshift` which moves\n            everything by this amount.\n        startarrowhead\n            Sets the start annotation arrow head style.\n        startarrowsize\n            Sets the size of the start annotation arrow head,\n            relative to `arrowwidth`. A value of 1 (default) gives\n            a head about 3x as wide as the line.\n        startstandoff\n            Sets a distance, in pixels, to move the start arrowhead\n            away from the position it is pointing at, for example\n            to point at the edge of a marker independent of zoom.\n            Note that this shortens the arrow from the `ax` / `ay`\n            vector, in contrast to `xshift` / `yshift` which moves\n            everything by this amount.\n        templateitemname\n            Used to refer to a named item in this array in the\n            template. Named items from the template will be created\n            even without a matching item in the input figure, but\n            you can modify one by making an item with\n            `templateitemname` matching its `name`, alongside your\n            modifications (including `visible: false` or `enabled:\n            false` to hide it). If there is no template or no\n            matching item, this item will be hidden unless you\n            explicitly show it with `visible: true`.\n        text\n            Sets the text associated with this annotation. Plotly\n            uses a subset of HTML tags to do things like newline\n            (<br>), bold (<b></b>), italics (<i></i>), hyperlinks\n            (<a href=\'...\'></a>). Tags <em>, <sup>, <sub> <span>\n            are also supported.\n        textangle\n            Sets the angle at which the `text` is drawn with\n            respect to the horizontal.\n        valign\n            Sets the vertical alignment of the `text` within the\n            box. Has an effect only if an explicit height is set to\n            override the text height.\n        visible\n            Determines whether or not this annotation is visible.\n        width\n            Sets an explicit width for the text box. null (default)\n            lets the text set the box width. Wider text will be\n            clipped. There is no automatic wrapping; use <br> to\n            start a new line.\n        x\n            Sets the annotation\'s x position. If the axis `type` is\n            "log", then you must take the log of your desired\n            range. If the axis `type` is "date", it should be date\n            strings, like date data, though Date objects and unix\n            milliseconds will be accepted and converted to strings.\n            If the axis `type` is "category", it should be numbers,\n            using the scale where each category is assigned a\n            serial number from zero in the order it appears.\n        xanchor\n            Sets the text box\'s horizontal position anchor This\n            anchor binds the `x` position to the "left", "center"\n            or "right" of the annotation. For example, if `x` is\n            set to 1, `xref` to "paper" and `xanchor` to "right"\n            then the right-most portion of the annotation lines up\n            with the right-most edge of the plotting area. If\n            "auto", the anchor is equivalent to "center" for data-\n            referenced annotations or if there is an arrow, whereas\n            for paper-referenced with no arrow, the anchor picked\n            corresponds to the closest side.\n        xclick\n            Toggle this annotation when clicking a data point whose\n            `x` value is `xclick` rather than the annotation\'s `x`\n            value.\n        xref\n            Sets the annotation\'s x coordinate axis. If set to a x\n            axis id (e.g. "x" or "x2"), the `x` position refers to\n            a x coordinate. If set to "paper", the `x` position\n            refers to the distance from the left of the plotting\n            area in normalized coordinates where 0 (1) corresponds\n            to the left (right). If set to a x axis ID followed by\n            "domain" (separated by a space), the position behaves\n            like for "paper", but refers to the distance in\n            fractions of the domain length from the left of the\n            domain of that axis: e.g., *x2 domain* refers to the\n            domain of the second x  axis and a x position of 0.5\n            refers to the point between the left and the right of\n            the domain of the second x axis.\n        xshift\n            Shifts the position of the whole annotation and arrow\n            to the right (positive) or left (negative) by this many\n            pixels.\n        y\n            Sets the annotation\'s y position. If the axis `type` is\n            "log", then you must take the log of your desired\n            range. If the axis `type` is "date", it should be date\n            strings, like date data, though Date objects and unix\n            milliseconds will be accepted and converted to strings.\n            If the axis `type` is "category", it should be numbers,\n            using the scale where each category is assigned a\n            serial number from zero in the order it appears.\n        yanchor\n            Sets the text box\'s vertical position anchor This\n            anchor binds the `y` position to the "top", "middle" or\n            "bottom" of the annotation. For example, if `y` is set\n            to 1, `yref` to "paper" and `yanchor` to "top" then the\n            top-most portion of the annotation lines up with the\n            top-most edge of the plotting area. If "auto", the\n            anchor is equivalent to "middle" for data-referenced\n            annotations or if there is an arrow, whereas for paper-\n            referenced with no arrow, the anchor picked corresponds\n            to the closest side.\n        yclick\n            Toggle this annotation when clicking a data point whose\n            `y` value is `yclick` rather than the annotation\'s `y`\n            value.\n        yref\n            Sets the annotation\'s y coordinate axis. If set to a y\n            axis id (e.g. "y" or "y2"), the `y` position refers to\n            a y coordinate. If set to "paper", the `y` position\n            refers to the distance from the bottom of the plotting\n            area in normalized coordinates where 0 (1) corresponds\n            to the bottom (top). If set to a y axis ID followed by\n            "domain" (separated by a space), the position behaves\n            like for "paper", but refers to the distance in\n            fractions of the domain length from the bottom of the\n            domain of that axis: e.g., *y2 domain* refers to the\n            domain of the second y  axis and a y position of 0.5\n            refers to the point between the bottom and the top of\n            the domain of the second y axis.\n        yshift\n            Shifts the position of the whole annotation and arrow\n            up (positive) or down (negative) by this many pixels.\n\n        Returns\n        -------\n        Annotation\n        '
        super(Annotation, self).__init__('annotations')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.Annotation\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.Annotation`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('align', None)
        _v = align if align is not None else _v
        if _v is not None:
            self['align'] = _v
        _v = arg.pop('arrowcolor', None)
        _v = arrowcolor if arrowcolor is not None else _v
        if _v is not None:
            self['arrowcolor'] = _v
        _v = arg.pop('arrowhead', None)
        _v = arrowhead if arrowhead is not None else _v
        if _v is not None:
            self['arrowhead'] = _v
        _v = arg.pop('arrowside', None)
        _v = arrowside if arrowside is not None else _v
        if _v is not None:
            self['arrowside'] = _v
        _v = arg.pop('arrowsize', None)
        _v = arrowsize if arrowsize is not None else _v
        if _v is not None:
            self['arrowsize'] = _v
        _v = arg.pop('arrowwidth', None)
        _v = arrowwidth if arrowwidth is not None else _v
        if _v is not None:
            self['arrowwidth'] = _v
        _v = arg.pop('ax', None)
        _v = ax if ax is not None else _v
        if _v is not None:
            self['ax'] = _v
        _v = arg.pop('axref', None)
        _v = axref if axref is not None else _v
        if _v is not None:
            self['axref'] = _v
        _v = arg.pop('ay', None)
        _v = ay if ay is not None else _v
        if _v is not None:
            self['ay'] = _v
        _v = arg.pop('ayref', None)
        _v = ayref if ayref is not None else _v
        if _v is not None:
            self['ayref'] = _v
        _v = arg.pop('bgcolor', None)
        _v = bgcolor if bgcolor is not None else _v
        if _v is not None:
            self['bgcolor'] = _v
        _v = arg.pop('bordercolor', None)
        _v = bordercolor if bordercolor is not None else _v
        if _v is not None:
            self['bordercolor'] = _v
        _v = arg.pop('borderpad', None)
        _v = borderpad if borderpad is not None else _v
        if _v is not None:
            self['borderpad'] = _v
        _v = arg.pop('borderwidth', None)
        _v = borderwidth if borderwidth is not None else _v
        if _v is not None:
            self['borderwidth'] = _v
        _v = arg.pop('captureevents', None)
        _v = captureevents if captureevents is not None else _v
        if _v is not None:
            self['captureevents'] = _v
        _v = arg.pop('clicktoshow', None)
        _v = clicktoshow if clicktoshow is not None else _v
        if _v is not None:
            self['clicktoshow'] = _v
        _v = arg.pop('font', None)
        _v = font if font is not None else _v
        if _v is not None:
            self['font'] = _v
        _v = arg.pop('height', None)
        _v = height if height is not None else _v
        if _v is not None:
            self['height'] = _v
        _v = arg.pop('hoverlabel', None)
        _v = hoverlabel if hoverlabel is not None else _v
        if _v is not None:
            self['hoverlabel'] = _v
        _v = arg.pop('hovertext', None)
        _v = hovertext if hovertext is not None else _v
        if _v is not None:
            self['hovertext'] = _v
        _v = arg.pop('name', None)
        _v = name if name is not None else _v
        if _v is not None:
            self['name'] = _v
        _v = arg.pop('opacity', None)
        _v = opacity if opacity is not None else _v
        if _v is not None:
            self['opacity'] = _v
        _v = arg.pop('showarrow', None)
        _v = showarrow if showarrow is not None else _v
        if _v is not None:
            self['showarrow'] = _v
        _v = arg.pop('standoff', None)
        _v = standoff if standoff is not None else _v
        if _v is not None:
            self['standoff'] = _v
        _v = arg.pop('startarrowhead', None)
        _v = startarrowhead if startarrowhead is not None else _v
        if _v is not None:
            self['startarrowhead'] = _v
        _v = arg.pop('startarrowsize', None)
        _v = startarrowsize if startarrowsize is not None else _v
        if _v is not None:
            self['startarrowsize'] = _v
        _v = arg.pop('startstandoff', None)
        _v = startstandoff if startstandoff is not None else _v
        if _v is not None:
            self['startstandoff'] = _v
        _v = arg.pop('templateitemname', None)
        _v = templateitemname if templateitemname is not None else _v
        if _v is not None:
            self['templateitemname'] = _v
        _v = arg.pop('text', None)
        _v = text if text is not None else _v
        if _v is not None:
            self['text'] = _v
        _v = arg.pop('textangle', None)
        _v = textangle if textangle is not None else _v
        if _v is not None:
            self['textangle'] = _v
        _v = arg.pop('valign', None)
        _v = valign if valign is not None else _v
        if _v is not None:
            self['valign'] = _v
        _v = arg.pop('visible', None)
        _v = visible if visible is not None else _v
        if _v is not None:
            self['visible'] = _v
        _v = arg.pop('width', None)
        _v = width if width is not None else _v
        if _v is not None:
            self['width'] = _v
        _v = arg.pop('x', None)
        _v = x if x is not None else _v
        if _v is not None:
            self['x'] = _v
        _v = arg.pop('xanchor', None)
        _v = xanchor if xanchor is not None else _v
        if _v is not None:
            self['xanchor'] = _v
        _v = arg.pop('xclick', None)
        _v = xclick if xclick is not None else _v
        if _v is not None:
            self['xclick'] = _v
        _v = arg.pop('xref', None)
        _v = xref if xref is not None else _v
        if _v is not None:
            self['xref'] = _v
        _v = arg.pop('xshift', None)
        _v = xshift if xshift is not None else _v
        if _v is not None:
            self['xshift'] = _v
        _v = arg.pop('y', None)
        _v = y if y is not None else _v
        if _v is not None:
            self['y'] = _v
        _v = arg.pop('yanchor', None)
        _v = yanchor if yanchor is not None else _v
        if _v is not None:
            self['yanchor'] = _v
        _v = arg.pop('yclick', None)
        _v = yclick if yclick is not None else _v
        if _v is not None:
            self['yclick'] = _v
        _v = arg.pop('yref', None)
        _v = yref if yref is not None else _v
        if _v is not None:
            self['yref'] = _v
        _v = arg.pop('yshift', None)
        _v = yshift if yshift is not None else _v
        if _v is not None:
            self['yshift'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False