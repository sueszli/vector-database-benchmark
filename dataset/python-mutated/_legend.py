from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Legend(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout'
    _path_str = 'layout.legend'
    _valid_props = {'bgcolor', 'bordercolor', 'borderwidth', 'entrywidth', 'entrywidthmode', 'font', 'groupclick', 'grouptitlefont', 'itemclick', 'itemdoubleclick', 'itemsizing', 'itemwidth', 'orientation', 'title', 'tracegroupgap', 'traceorder', 'uirevision', 'valign', 'visible', 'x', 'xanchor', 'xref', 'y', 'yanchor', 'yref'}

    @property
    def bgcolor(self):
        if False:
            return 10
        "\n        Sets the legend background color. Defaults to\n        `layout.paper_bgcolor`.\n\n        The 'bgcolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['bgcolor']

    @bgcolor.setter
    def bgcolor(self, val):
        if False:
            while True:
                i = 10
        self['bgcolor'] = val

    @property
    def bordercolor(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the color of the border enclosing the legend.\n\n        The 'bordercolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
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
            print('Hello World!')
        "\n        Sets the width (in px) of the border enclosing the legend.\n\n        The 'borderwidth' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['borderwidth']

    @borderwidth.setter
    def borderwidth(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['borderwidth'] = val

    @property
    def entrywidth(self):
        if False:
            while True:
                i = 10
        '\n        Sets the width (in px or fraction) of the legend. Use 0 to size\n        the entry based on the text width, when `entrywidthmode` is set\n        to "pixels".\n\n        The \'entrywidth\' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        '
        return self['entrywidth']

    @entrywidth.setter
    def entrywidth(self, val):
        if False:
            print('Hello World!')
        self['entrywidth'] = val

    @property
    def entrywidthmode(self):
        if False:
            return 10
        "\n        Determines what entrywidth means.\n\n        The 'entrywidthmode' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['fraction', 'pixels']\n\n        Returns\n        -------\n        Any\n        "
        return self['entrywidthmode']

    @entrywidthmode.setter
    def entrywidthmode(self, val):
        if False:
            i = 10
            return i + 15
        self['entrywidthmode'] = val

    @property
    def font(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the font used to text the legend items.\n\n        The \'font\' property is an instance of Font\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.legend.Font`\n          - A dict of string/value properties that will be passed\n            to the Font constructor\n\n            Supported dict properties:\n\n                color\n\n                family\n                    HTML font family - the typeface that will be\n                    applied by the web browser. The web browser\n                    will only be able to apply a font if it is\n                    available on the system which it operates.\n                    Provide multiple font families, separated by\n                    commas, to indicate the preference in which to\n                    apply fonts if they aren\'t available on the\n                    system. The Chart Studio Cloud (at\n                    https://chart-studio.plotly.com or on-premise)\n                    generates images on a server, where only a\n                    select number of fonts are installed and\n                    supported. These include "Arial", "Balto",\n                    "Courier New", "Droid Sans",, "Droid Serif",\n                    "Droid Sans Mono", "Gravitas One", "Old\n                    Standard TT", "Open Sans", "Overpass", "PT Sans\n                    Narrow", "Raleway", "Times New Roman".\n                size\n\n        Returns\n        -------\n        plotly.graph_objs.layout.legend.Font\n        '
        return self['font']

    @font.setter
    def font(self, val):
        if False:
            i = 10
            return i + 15
        self['font'] = val

    @property
    def groupclick(self):
        if False:
            return 10
        '\n        Determines the behavior on legend group item click.\n        "toggleitem" toggles the visibility of the individual item\n        clicked on the graph. "togglegroup" toggles the visibility of\n        all items in the same legendgroup as the item clicked on the\n        graph.\n\n        The \'groupclick\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'toggleitem\', \'togglegroup\']\n\n        Returns\n        -------\n        Any\n        '
        return self['groupclick']

    @groupclick.setter
    def groupclick(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['groupclick'] = val

    @property
    def grouptitlefont(self):
        if False:
            print('Hello World!')
        '\n        Sets the font for group titles in legend. Defaults to\n        `legend.font` with its size increased about 10%.\n\n        The \'grouptitlefont\' property is an instance of Grouptitlefont\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.legend.Grouptitlefont`\n          - A dict of string/value properties that will be passed\n            to the Grouptitlefont constructor\n\n            Supported dict properties:\n\n                color\n\n                family\n                    HTML font family - the typeface that will be\n                    applied by the web browser. The web browser\n                    will only be able to apply a font if it is\n                    available on the system which it operates.\n                    Provide multiple font families, separated by\n                    commas, to indicate the preference in which to\n                    apply fonts if they aren\'t available on the\n                    system. The Chart Studio Cloud (at\n                    https://chart-studio.plotly.com or on-premise)\n                    generates images on a server, where only a\n                    select number of fonts are installed and\n                    supported. These include "Arial", "Balto",\n                    "Courier New", "Droid Sans",, "Droid Serif",\n                    "Droid Sans Mono", "Gravitas One", "Old\n                    Standard TT", "Open Sans", "Overpass", "PT Sans\n                    Narrow", "Raleway", "Times New Roman".\n                size\n\n        Returns\n        -------\n        plotly.graph_objs.layout.legend.Grouptitlefont\n        '
        return self['grouptitlefont']

    @grouptitlefont.setter
    def grouptitlefont(self, val):
        if False:
            print('Hello World!')
        self['grouptitlefont'] = val

    @property
    def itemclick(self):
        if False:
            i = 10
            return i + 15
        '\n        Determines the behavior on legend item click. "toggle" toggles\n        the visibility of the item clicked on the graph. "toggleothers"\n        makes the clicked item the sole visible item on the graph.\n        False disables legend item click interactions.\n\n        The \'itemclick\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'toggle\', \'toggleothers\', False]\n\n        Returns\n        -------\n        Any\n        '
        return self['itemclick']

    @itemclick.setter
    def itemclick(self, val):
        if False:
            while True:
                i = 10
        self['itemclick'] = val

    @property
    def itemdoubleclick(self):
        if False:
            while True:
                i = 10
        '\n        Determines the behavior on legend item double-click. "toggle"\n        toggles the visibility of the item clicked on the graph.\n        "toggleothers" makes the clicked item the sole visible item on\n        the graph. False disables legend item double-click\n        interactions.\n\n        The \'itemdoubleclick\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'toggle\', \'toggleothers\', False]\n\n        Returns\n        -------\n        Any\n        '
        return self['itemdoubleclick']

    @itemdoubleclick.setter
    def itemdoubleclick(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['itemdoubleclick'] = val

    @property
    def itemsizing(self):
        if False:
            i = 10
            return i + 15
        '\n        Determines if the legend items symbols scale with their\n        corresponding "trace" attributes or remain "constant"\n        independent of the symbol size on the graph.\n\n        The \'itemsizing\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'trace\', \'constant\']\n\n        Returns\n        -------\n        Any\n        '
        return self['itemsizing']

    @itemsizing.setter
    def itemsizing(self, val):
        if False:
            print('Hello World!')
        self['itemsizing'] = val

    @property
    def itemwidth(self):
        if False:
            return 10
        "\n        Sets the width (in px) of the legend item symbols (the part\n        other than the title.text).\n\n        The 'itemwidth' property is a number and may be specified as:\n          - An int or float in the interval [30, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['itemwidth']

    @itemwidth.setter
    def itemwidth(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['itemwidth'] = val

    @property
    def orientation(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the orientation of the legend.\n\n        The 'orientation' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['v', 'h']\n\n        Returns\n        -------\n        Any\n        "
        return self['orientation']

    @orientation.setter
    def orientation(self, val):
        if False:
            return 10
        self['orientation'] = val

    @property
    def title(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The \'title\' property is an instance of Title\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.legend.Title`\n          - A dict of string/value properties that will be passed\n            to the Title constructor\n\n            Supported dict properties:\n\n                font\n                    Sets this legend\'s title font. Defaults to\n                    `legend.font` with its size increased about\n                    20%.\n                side\n                    Determines the location of legend\'s title with\n                    respect to the legend items. Defaulted to "top"\n                    with `orientation` is "h". Defaulted to "left"\n                    with `orientation` is "v". The *top left*\n                    options could be used to expand top center and\n                    top right are for horizontal alignment legend\n                    area in both x and y sides.\n                text\n                    Sets the title of the legend.\n\n        Returns\n        -------\n        plotly.graph_objs.layout.legend.Title\n        '
        return self['title']

    @title.setter
    def title(self, val):
        if False:
            while True:
                i = 10
        self['title'] = val

    @property
    def tracegroupgap(self):
        if False:
            while True:
                i = 10
        "\n        Sets the amount of vertical space (in px) between legend\n        groups.\n\n        The 'tracegroupgap' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['tracegroupgap']

    @tracegroupgap.setter
    def tracegroupgap(self, val):
        if False:
            return 10
        self['tracegroupgap'] = val

    @property
    def traceorder(self):
        if False:
            print('Hello World!')
        '\n        Determines the order at which the legend items are displayed.\n        If "normal", the items are displayed top-to-bottom in the same\n        order as the input data. If "reversed", the items are displayed\n        in the opposite order as "normal". If "grouped", the items are\n        displayed in groups (when a trace `legendgroup` is provided).\n        if "grouped+reversed", the items are displayed in the opposite\n        order as "grouped".\n\n        The \'traceorder\' property is a flaglist and may be specified\n        as a string containing:\n          - Any combination of [\'reversed\', \'grouped\'] joined with \'+\' characters\n            (e.g. \'reversed+grouped\')\n            OR exactly one of [\'normal\'] (e.g. \'normal\')\n\n        Returns\n        -------\n        Any\n        '
        return self['traceorder']

    @traceorder.setter
    def traceorder(self, val):
        if False:
            return 10
        self['traceorder'] = val

    @property
    def uirevision(self):
        if False:
            i = 10
            return i + 15
        "\n        Controls persistence of legend-driven changes in trace and pie\n        label visibility. Defaults to `layout.uirevision`.\n\n        The 'uirevision' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['uirevision']

    @uirevision.setter
    def uirevision(self, val):
        if False:
            print('Hello World!')
        self['uirevision'] = val

    @property
    def valign(self):
        if False:
            return 10
        "\n        Sets the vertical alignment of the symbols with respect to\n        their associated text.\n\n        The 'valign' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['top', 'middle', 'bottom']\n\n        Returns\n        -------\n        Any\n        "
        return self['valign']

    @valign.setter
    def valign(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['valign'] = val

    @property
    def visible(self):
        if False:
            i = 10
            return i + 15
        "\n        Determines whether or not this legend is visible.\n\n        The 'visible' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['visible']

    @visible.setter
    def visible(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['visible'] = val

    @property
    def x(self):
        if False:
            print('Hello World!')
        '\n        Sets the x position with respect to `xref` (in normalized\n        coordinates) of the legend. When `xref` is "paper", defaults to\n        1.02 for vertical legends and defaults to 0 for horizontal\n        legends. When `xref` is "container", defaults to 1 for vertical\n        legends and defaults to 0 for horizontal legends. Must be\n        between 0 and 1 if `xref` is "container". and between "-2" and\n        3 if `xref` is "paper".\n\n        The \'x\' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        '
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
            i = 10
            return i + 15
        '\n        Sets the legend\'s horizontal position anchor. This anchor binds\n        the `x` position to the "left", "center" or "right" of the\n        legend. Value "auto" anchors legends to the right for `x`\n        values greater than or equal to 2/3, anchors legends to the\n        left for `x` values less than or equal to 1/3 and anchors\n        legends with respect to their center otherwise.\n\n        The \'xanchor\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'auto\', \'left\', \'center\', \'right\']\n\n        Returns\n        -------\n        Any\n        '
        return self['xanchor']

    @xanchor.setter
    def xanchor(self, val):
        if False:
            i = 10
            return i + 15
        self['xanchor'] = val

    @property
    def xref(self):
        if False:
            print('Hello World!')
        '\n        Sets the container `x` refers to. "container" spans the entire\n        `width` of the plot. "paper" refers to the width of the\n        plotting area only.\n\n        The \'xref\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'container\', \'paper\']\n\n        Returns\n        -------\n        Any\n        '
        return self['xref']

    @xref.setter
    def xref(self, val):
        if False:
            print('Hello World!')
        self['xref'] = val

    @property
    def y(self):
        if False:
            print('Hello World!')
        '\n        Sets the y position with respect to `yref` (in normalized\n        coordinates) of the legend. When `yref` is "paper", defaults to\n        1 for vertical legends, defaults to "-0.1" for horizontal\n        legends on graphs w/o range sliders and defaults to 1.1 for\n        horizontal legends on graph with one or multiple range sliders.\n        When `yref` is "container", defaults to 1. Must be between 0\n        and 1 if `yref` is "container" and between "-2" and 3 if `yref`\n        is "paper".\n\n        The \'y\' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        '
        return self['y']

    @y.setter
    def y(self, val):
        if False:
            while True:
                i = 10
        self['y'] = val

    @property
    def yanchor(self):
        if False:
            while True:
                i = 10
        '\n        Sets the legend\'s vertical position anchor This anchor binds\n        the `y` position to the "top", "middle" or "bottom" of the\n        legend. Value "auto" anchors legends at their bottom for `y`\n        values less than or equal to 1/3, anchors legends to at their\n        top for `y` values greater than or equal to 2/3 and anchors\n        legends with respect to their middle otherwise.\n\n        The \'yanchor\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'auto\', \'top\', \'middle\', \'bottom\']\n\n        Returns\n        -------\n        Any\n        '
        return self['yanchor']

    @yanchor.setter
    def yanchor(self, val):
        if False:
            while True:
                i = 10
        self['yanchor'] = val

    @property
    def yref(self):
        if False:
            print('Hello World!')
        '\n        Sets the container `y` refers to. "container" spans the entire\n        `height` of the plot. "paper" refers to the height of the\n        plotting area only.\n\n        The \'yref\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'container\', \'paper\']\n\n        Returns\n        -------\n        Any\n        '
        return self['yref']

    @yref.setter
    def yref(self, val):
        if False:
            i = 10
            return i + 15
        self['yref'] = val

    @property
    def _prop_descriptions(self):
        if False:
            while True:
                i = 10
        return '        bgcolor\n            Sets the legend background color. Defaults to\n            `layout.paper_bgcolor`.\n        bordercolor\n            Sets the color of the border enclosing the legend.\n        borderwidth\n            Sets the width (in px) of the border enclosing the\n            legend.\n        entrywidth\n            Sets the width (in px or fraction) of the legend. Use 0\n            to size the entry based on the text width, when\n            `entrywidthmode` is set to "pixels".\n        entrywidthmode\n            Determines what entrywidth means.\n        font\n            Sets the font used to text the legend items.\n        groupclick\n            Determines the behavior on legend group item click.\n            "toggleitem" toggles the visibility of the individual\n            item clicked on the graph. "togglegroup" toggles the\n            visibility of all items in the same legendgroup as the\n            item clicked on the graph.\n        grouptitlefont\n            Sets the font for group titles in legend. Defaults to\n            `legend.font` with its size increased about 10%.\n        itemclick\n            Determines the behavior on legend item click. "toggle"\n            toggles the visibility of the item clicked on the\n            graph. "toggleothers" makes the clicked item the sole\n            visible item on the graph. False disables legend item\n            click interactions.\n        itemdoubleclick\n            Determines the behavior on legend item double-click.\n            "toggle" toggles the visibility of the item clicked on\n            the graph. "toggleothers" makes the clicked item the\n            sole visible item on the graph. False disables legend\n            item double-click interactions.\n        itemsizing\n            Determines if the legend items symbols scale with their\n            corresponding "trace" attributes or remain "constant"\n            independent of the symbol size on the graph.\n        itemwidth\n            Sets the width (in px) of the legend item symbols (the\n            part other than the title.text).\n        orientation\n            Sets the orientation of the legend.\n        title\n            :class:`plotly.graph_objects.layout.legend.Title`\n            instance or dict with compatible properties\n        tracegroupgap\n            Sets the amount of vertical space (in px) between\n            legend groups.\n        traceorder\n            Determines the order at which the legend items are\n            displayed. If "normal", the items are displayed top-to-\n            bottom in the same order as the input data. If\n            "reversed", the items are displayed in the opposite\n            order as "normal". If "grouped", the items are\n            displayed in groups (when a trace `legendgroup` is\n            provided). if "grouped+reversed", the items are\n            displayed in the opposite order as "grouped".\n        uirevision\n            Controls persistence of legend-driven changes in trace\n            and pie label visibility. Defaults to\n            `layout.uirevision`.\n        valign\n            Sets the vertical alignment of the symbols with respect\n            to their associated text.\n        visible\n            Determines whether or not this legend is visible.\n        x\n            Sets the x position with respect to `xref` (in\n            normalized coordinates) of the legend. When `xref` is\n            "paper", defaults to 1.02 for vertical legends and\n            defaults to 0 for horizontal legends. When `xref` is\n            "container", defaults to 1 for vertical legends and\n            defaults to 0 for horizontal legends. Must be between 0\n            and 1 if `xref` is "container". and between "-2" and 3\n            if `xref` is "paper".\n        xanchor\n            Sets the legend\'s horizontal position anchor. This\n            anchor binds the `x` position to the "left", "center"\n            or "right" of the legend. Value "auto" anchors legends\n            to the right for `x` values greater than or equal to\n            2/3, anchors legends to the left for `x` values less\n            than or equal to 1/3 and anchors legends with respect\n            to their center otherwise.\n        xref\n            Sets the container `x` refers to. "container" spans the\n            entire `width` of the plot. "paper" refers to the width\n            of the plotting area only.\n        y\n            Sets the y position with respect to `yref` (in\n            normalized coordinates) of the legend. When `yref` is\n            "paper", defaults to 1 for vertical legends, defaults\n            to "-0.1" for horizontal legends on graphs w/o range\n            sliders and defaults to 1.1 for horizontal legends on\n            graph with one or multiple range sliders. When `yref`\n            is "container", defaults to 1. Must be between 0 and 1\n            if `yref` is "container" and between "-2" and 3 if\n            `yref` is "paper".\n        yanchor\n            Sets the legend\'s vertical position anchor This anchor\n            binds the `y` position to the "top", "middle" or\n            "bottom" of the legend. Value "auto" anchors legends at\n            their bottom for `y` values less than or equal to 1/3,\n            anchors legends to at their top for `y` values greater\n            than or equal to 2/3 and anchors legends with respect\n            to their middle otherwise.\n        yref\n            Sets the container `y` refers to. "container" spans the\n            entire `height` of the plot. "paper" refers to the\n            height of the plotting area only.\n        '

    def __init__(self, arg=None, bgcolor=None, bordercolor=None, borderwidth=None, entrywidth=None, entrywidthmode=None, font=None, groupclick=None, grouptitlefont=None, itemclick=None, itemdoubleclick=None, itemsizing=None, itemwidth=None, orientation=None, title=None, tracegroupgap=None, traceorder=None, uirevision=None, valign=None, visible=None, x=None, xanchor=None, xref=None, y=None, yanchor=None, yref=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a new Legend object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of :class:`plotly.graph_objs.layout.Legend`\n        bgcolor\n            Sets the legend background color. Defaults to\n            `layout.paper_bgcolor`.\n        bordercolor\n            Sets the color of the border enclosing the legend.\n        borderwidth\n            Sets the width (in px) of the border enclosing the\n            legend.\n        entrywidth\n            Sets the width (in px or fraction) of the legend. Use 0\n            to size the entry based on the text width, when\n            `entrywidthmode` is set to "pixels".\n        entrywidthmode\n            Determines what entrywidth means.\n        font\n            Sets the font used to text the legend items.\n        groupclick\n            Determines the behavior on legend group item click.\n            "toggleitem" toggles the visibility of the individual\n            item clicked on the graph. "togglegroup" toggles the\n            visibility of all items in the same legendgroup as the\n            item clicked on the graph.\n        grouptitlefont\n            Sets the font for group titles in legend. Defaults to\n            `legend.font` with its size increased about 10%.\n        itemclick\n            Determines the behavior on legend item click. "toggle"\n            toggles the visibility of the item clicked on the\n            graph. "toggleothers" makes the clicked item the sole\n            visible item on the graph. False disables legend item\n            click interactions.\n        itemdoubleclick\n            Determines the behavior on legend item double-click.\n            "toggle" toggles the visibility of the item clicked on\n            the graph. "toggleothers" makes the clicked item the\n            sole visible item on the graph. False disables legend\n            item double-click interactions.\n        itemsizing\n            Determines if the legend items symbols scale with their\n            corresponding "trace" attributes or remain "constant"\n            independent of the symbol size on the graph.\n        itemwidth\n            Sets the width (in px) of the legend item symbols (the\n            part other than the title.text).\n        orientation\n            Sets the orientation of the legend.\n        title\n            :class:`plotly.graph_objects.layout.legend.Title`\n            instance or dict with compatible properties\n        tracegroupgap\n            Sets the amount of vertical space (in px) between\n            legend groups.\n        traceorder\n            Determines the order at which the legend items are\n            displayed. If "normal", the items are displayed top-to-\n            bottom in the same order as the input data. If\n            "reversed", the items are displayed in the opposite\n            order as "normal". If "grouped", the items are\n            displayed in groups (when a trace `legendgroup` is\n            provided). if "grouped+reversed", the items are\n            displayed in the opposite order as "grouped".\n        uirevision\n            Controls persistence of legend-driven changes in trace\n            and pie label visibility. Defaults to\n            `layout.uirevision`.\n        valign\n            Sets the vertical alignment of the symbols with respect\n            to their associated text.\n        visible\n            Determines whether or not this legend is visible.\n        x\n            Sets the x position with respect to `xref` (in\n            normalized coordinates) of the legend. When `xref` is\n            "paper", defaults to 1.02 for vertical legends and\n            defaults to 0 for horizontal legends. When `xref` is\n            "container", defaults to 1 for vertical legends and\n            defaults to 0 for horizontal legends. Must be between 0\n            and 1 if `xref` is "container". and between "-2" and 3\n            if `xref` is "paper".\n        xanchor\n            Sets the legend\'s horizontal position anchor. This\n            anchor binds the `x` position to the "left", "center"\n            or "right" of the legend. Value "auto" anchors legends\n            to the right for `x` values greater than or equal to\n            2/3, anchors legends to the left for `x` values less\n            than or equal to 1/3 and anchors legends with respect\n            to their center otherwise.\n        xref\n            Sets the container `x` refers to. "container" spans the\n            entire `width` of the plot. "paper" refers to the width\n            of the plotting area only.\n        y\n            Sets the y position with respect to `yref` (in\n            normalized coordinates) of the legend. When `yref` is\n            "paper", defaults to 1 for vertical legends, defaults\n            to "-0.1" for horizontal legends on graphs w/o range\n            sliders and defaults to 1.1 for horizontal legends on\n            graph with one or multiple range sliders. When `yref`\n            is "container", defaults to 1. Must be between 0 and 1\n            if `yref` is "container" and between "-2" and 3 if\n            `yref` is "paper".\n        yanchor\n            Sets the legend\'s vertical position anchor This anchor\n            binds the `y` position to the "top", "middle" or\n            "bottom" of the legend. Value "auto" anchors legends at\n            their bottom for `y` values less than or equal to 1/3,\n            anchors legends to at their top for `y` values greater\n            than or equal to 2/3 and anchors legends with respect\n            to their middle otherwise.\n        yref\n            Sets the container `y` refers to. "container" spans the\n            entire `height` of the plot. "paper" refers to the\n            height of the plotting area only.\n\n        Returns\n        -------\n        Legend\n        '
        super(Legend, self).__init__('legend')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.Legend\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.Legend`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
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
        _v = arg.pop('entrywidth', None)
        _v = entrywidth if entrywidth is not None else _v
        if _v is not None:
            self['entrywidth'] = _v
        _v = arg.pop('entrywidthmode', None)
        _v = entrywidthmode if entrywidthmode is not None else _v
        if _v is not None:
            self['entrywidthmode'] = _v
        _v = arg.pop('font', None)
        _v = font if font is not None else _v
        if _v is not None:
            self['font'] = _v
        _v = arg.pop('groupclick', None)
        _v = groupclick if groupclick is not None else _v
        if _v is not None:
            self['groupclick'] = _v
        _v = arg.pop('grouptitlefont', None)
        _v = grouptitlefont if grouptitlefont is not None else _v
        if _v is not None:
            self['grouptitlefont'] = _v
        _v = arg.pop('itemclick', None)
        _v = itemclick if itemclick is not None else _v
        if _v is not None:
            self['itemclick'] = _v
        _v = arg.pop('itemdoubleclick', None)
        _v = itemdoubleclick if itemdoubleclick is not None else _v
        if _v is not None:
            self['itemdoubleclick'] = _v
        _v = arg.pop('itemsizing', None)
        _v = itemsizing if itemsizing is not None else _v
        if _v is not None:
            self['itemsizing'] = _v
        _v = arg.pop('itemwidth', None)
        _v = itemwidth if itemwidth is not None else _v
        if _v is not None:
            self['itemwidth'] = _v
        _v = arg.pop('orientation', None)
        _v = orientation if orientation is not None else _v
        if _v is not None:
            self['orientation'] = _v
        _v = arg.pop('title', None)
        _v = title if title is not None else _v
        if _v is not None:
            self['title'] = _v
        _v = arg.pop('tracegroupgap', None)
        _v = tracegroupgap if tracegroupgap is not None else _v
        if _v is not None:
            self['tracegroupgap'] = _v
        _v = arg.pop('traceorder', None)
        _v = traceorder if traceorder is not None else _v
        if _v is not None:
            self['traceorder'] = _v
        _v = arg.pop('uirevision', None)
        _v = uirevision if uirevision is not None else _v
        if _v is not None:
            self['uirevision'] = _v
        _v = arg.pop('valign', None)
        _v = valign if valign is not None else _v
        if _v is not None:
            self['valign'] = _v
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
        _v = arg.pop('xref', None)
        _v = xref if xref is not None else _v
        if _v is not None:
            self['xref'] = _v
        _v = arg.pop('y', None)
        _v = y if y is not None else _v
        if _v is not None:
            self['y'] = _v
        _v = arg.pop('yanchor', None)
        _v = yanchor if yanchor is not None else _v
        if _v is not None:
            self['yanchor'] = _v
        _v = arg.pop('yref', None)
        _v = yref if yref is not None else _v
        if _v is not None:
            self['yref'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False