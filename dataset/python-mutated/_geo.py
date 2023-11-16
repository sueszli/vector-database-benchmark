from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Geo(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout'
    _path_str = 'layout.geo'
    _valid_props = {'bgcolor', 'center', 'coastlinecolor', 'coastlinewidth', 'countrycolor', 'countrywidth', 'domain', 'fitbounds', 'framecolor', 'framewidth', 'lakecolor', 'landcolor', 'lataxis', 'lonaxis', 'oceancolor', 'projection', 'resolution', 'rivercolor', 'riverwidth', 'scope', 'showcoastlines', 'showcountries', 'showframe', 'showlakes', 'showland', 'showocean', 'showrivers', 'showsubunits', 'subunitcolor', 'subunitwidth', 'uirevision', 'visible'}

    @property
    def bgcolor(self):
        if False:
            return 10
        "\n        Set the background color of the map\n\n        The 'bgcolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['bgcolor']

    @bgcolor.setter
    def bgcolor(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['bgcolor'] = val

    @property
    def center(self):
        if False:
            return 10
        "\n        The 'center' property is an instance of Center\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.geo.Center`\n          - A dict of string/value properties that will be passed\n            to the Center constructor\n\n            Supported dict properties:\n\n                lat\n                    Sets the latitude of the map's center. For all\n                    projection types, the map's latitude center\n                    lies at the middle of the latitude range by\n                    default.\n                lon\n                    Sets the longitude of the map's center. By\n                    default, the map's longitude center lies at the\n                    middle of the longitude range for scoped\n                    projection and above `projection.rotation.lon`\n                    otherwise.\n\n        Returns\n        -------\n        plotly.graph_objs.layout.geo.Center\n        "
        return self['center']

    @center.setter
    def center(self, val):
        if False:
            i = 10
            return i + 15
        self['center'] = val

    @property
    def coastlinecolor(self):
        if False:
            return 10
        "\n        Sets the coastline color.\n\n        The 'coastlinecolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['coastlinecolor']

    @coastlinecolor.setter
    def coastlinecolor(self, val):
        if False:
            return 10
        self['coastlinecolor'] = val

    @property
    def coastlinewidth(self):
        if False:
            return 10
        "\n        Sets the coastline stroke width (in px).\n\n        The 'coastlinewidth' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['coastlinewidth']

    @coastlinewidth.setter
    def coastlinewidth(self, val):
        if False:
            print('Hello World!')
        self['coastlinewidth'] = val

    @property
    def countrycolor(self):
        if False:
            return 10
        "\n        Sets line color of the country boundaries.\n\n        The 'countrycolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['countrycolor']

    @countrycolor.setter
    def countrycolor(self, val):
        if False:
            return 10
        self['countrycolor'] = val

    @property
    def countrywidth(self):
        if False:
            return 10
        "\n        Sets line width (in px) of the country boundaries.\n\n        The 'countrywidth' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['countrywidth']

    @countrywidth.setter
    def countrywidth(self, val):
        if False:
            i = 10
            return i + 15
        self['countrywidth'] = val

    @property
    def domain(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The 'domain' property is an instance of Domain\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.geo.Domain`\n          - A dict of string/value properties that will be passed\n            to the Domain constructor\n\n            Supported dict properties:\n\n                column\n                    If there is a layout grid, use the domain for\n                    this column in the grid for this geo subplot .\n                    Note that geo subplots are constrained by\n                    domain. In general, when `projection.scale` is\n                    set to 1. a map will fit either its x or y\n                    domain, but not both.\n                row\n                    If there is a layout grid, use the domain for\n                    this row in the grid for this geo subplot .\n                    Note that geo subplots are constrained by\n                    domain. In general, when `projection.scale` is\n                    set to 1. a map will fit either its x or y\n                    domain, but not both.\n                x\n                    Sets the horizontal domain of this geo subplot\n                    (in plot fraction). Note that geo subplots are\n                    constrained by domain. In general, when\n                    `projection.scale` is set to 1. a map will fit\n                    either its x or y domain, but not both.\n                y\n                    Sets the vertical domain of this geo subplot\n                    (in plot fraction). Note that geo subplots are\n                    constrained by domain. In general, when\n                    `projection.scale` is set to 1. a map will fit\n                    either its x or y domain, but not both.\n\n        Returns\n        -------\n        plotly.graph_objs.layout.geo.Domain\n        "
        return self['domain']

    @domain.setter
    def domain(self, val):
        if False:
            while True:
                i = 10
        self['domain'] = val

    @property
    def fitbounds(self):
        if False:
            print('Hello World!')
        '\n        Determines if this subplot\'s view settings are auto-computed to\n        fit trace data. On scoped maps, setting `fitbounds` leads to\n        `center.lon` and `center.lat` getting auto-filled. On maps with\n        a non-clipped projection, setting `fitbounds` leads to\n        `center.lon`, `center.lat`, and `projection.rotation.lon`\n        getting auto-filled. On maps with a clipped projection, setting\n        `fitbounds` leads to `center.lon`, `center.lat`,\n        `projection.rotation.lon`, `projection.rotation.lat`,\n        `lonaxis.range` and `lonaxis.range` getting auto-filled. If\n        "locations", only the trace\'s visible locations are considered\n        in the `fitbounds` computations. If "geojson", the entire trace\n        input `geojson` (if provided) is considered in the `fitbounds`\n        computations, Defaults to False.\n\n        The \'fitbounds\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [False, \'locations\', \'geojson\']\n\n        Returns\n        -------\n        Any\n        '
        return self['fitbounds']

    @fitbounds.setter
    def fitbounds(self, val):
        if False:
            print('Hello World!')
        self['fitbounds'] = val

    @property
    def framecolor(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the color the frame.\n\n        The 'framecolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['framecolor']

    @framecolor.setter
    def framecolor(self, val):
        if False:
            print('Hello World!')
        self['framecolor'] = val

    @property
    def framewidth(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the stroke width (in px) of the frame.\n\n        The 'framewidth' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['framewidth']

    @framewidth.setter
    def framewidth(self, val):
        if False:
            print('Hello World!')
        self['framewidth'] = val

    @property
    def lakecolor(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the color of the lakes.\n\n        The 'lakecolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['lakecolor']

    @lakecolor.setter
    def lakecolor(self, val):
        if False:
            print('Hello World!')
        self['lakecolor'] = val

    @property
    def landcolor(self):
        if False:
            return 10
        "\n        Sets the land mass color.\n\n        The 'landcolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['landcolor']

    @landcolor.setter
    def landcolor(self, val):
        if False:
            print('Hello World!')
        self['landcolor'] = val

    @property
    def lataxis(self):
        if False:
            i = 10
            return i + 15
        '\n        The \'lataxis\' property is an instance of Lataxis\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.geo.Lataxis`\n          - A dict of string/value properties that will be passed\n            to the Lataxis constructor\n\n            Supported dict properties:\n\n                dtick\n                    Sets the graticule\'s longitude/latitude tick\n                    step.\n                gridcolor\n                    Sets the graticule\'s stroke color.\n                griddash\n                    Sets the dash style of lines. Set to a dash\n                    type string ("solid", "dot", "dash",\n                    "longdash", "dashdot", or "longdashdot") or a\n                    dash length list in px (eg "5px,10px,2px,2px").\n                gridwidth\n                    Sets the graticule\'s stroke width (in px).\n                range\n                    Sets the range of this axis (in degrees), sets\n                    the map\'s clipped coordinates.\n                showgrid\n                    Sets whether or not graticule are shown on the\n                    map.\n                tick0\n                    Sets the graticule\'s starting tick\n                    longitude/latitude.\n\n        Returns\n        -------\n        plotly.graph_objs.layout.geo.Lataxis\n        '
        return self['lataxis']

    @lataxis.setter
    def lataxis(self, val):
        if False:
            i = 10
            return i + 15
        self['lataxis'] = val

    @property
    def lonaxis(self):
        if False:
            print('Hello World!')
        '\n        The \'lonaxis\' property is an instance of Lonaxis\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.geo.Lonaxis`\n          - A dict of string/value properties that will be passed\n            to the Lonaxis constructor\n\n            Supported dict properties:\n\n                dtick\n                    Sets the graticule\'s longitude/latitude tick\n                    step.\n                gridcolor\n                    Sets the graticule\'s stroke color.\n                griddash\n                    Sets the dash style of lines. Set to a dash\n                    type string ("solid", "dot", "dash",\n                    "longdash", "dashdot", or "longdashdot") or a\n                    dash length list in px (eg "5px,10px,2px,2px").\n                gridwidth\n                    Sets the graticule\'s stroke width (in px).\n                range\n                    Sets the range of this axis (in degrees), sets\n                    the map\'s clipped coordinates.\n                showgrid\n                    Sets whether or not graticule are shown on the\n                    map.\n                tick0\n                    Sets the graticule\'s starting tick\n                    longitude/latitude.\n\n        Returns\n        -------\n        plotly.graph_objs.layout.geo.Lonaxis\n        '
        return self['lonaxis']

    @lonaxis.setter
    def lonaxis(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['lonaxis'] = val

    @property
    def oceancolor(self):
        if False:
            print('Hello World!')
        "\n        Sets the ocean color\n\n        The 'oceancolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['oceancolor']

    @oceancolor.setter
    def oceancolor(self, val):
        if False:
            return 10
        self['oceancolor'] = val

    @property
    def projection(self):
        if False:
            i = 10
            return i + 15
        "\n        The 'projection' property is an instance of Projection\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.geo.Projection`\n          - A dict of string/value properties that will be passed\n            to the Projection constructor\n\n            Supported dict properties:\n\n                distance\n                    For satellite projection type only. Sets the\n                    distance from the center of the sphere to the\n                    point of view as a proportion of the sphere’s\n                    radius.\n                parallels\n                    For conic projection types only. Sets the\n                    parallels (tangent, secant) where the cone\n                    intersects the sphere.\n                rotation\n                    :class:`plotly.graph_objects.layout.geo.project\n                    ion.Rotation` instance or dict with compatible\n                    properties\n                scale\n                    Zooms in or out on the map view. A scale of 1\n                    corresponds to the largest zoom level that fits\n                    the map's lon and lat ranges.\n                tilt\n                    For satellite projection type only. Sets the\n                    tilt angle of perspective projection.\n                type\n                    Sets the projection type.\n\n        Returns\n        -------\n        plotly.graph_objs.layout.geo.Projection\n        "
        return self['projection']

    @projection.setter
    def projection(self, val):
        if False:
            while True:
                i = 10
        self['projection'] = val

    @property
    def resolution(self):
        if False:
            print('Hello World!')
        "\n        Sets the resolution of the base layers. The values have units\n        of km/mm e.g. 110 corresponds to a scale ratio of\n        1:110,000,000.\n\n        The 'resolution' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [110, 50]\n\n        Returns\n        -------\n        Any\n        "
        return self['resolution']

    @resolution.setter
    def resolution(self, val):
        if False:
            while True:
                i = 10
        self['resolution'] = val

    @property
    def rivercolor(self):
        if False:
            print('Hello World!')
        "\n        Sets color of the rivers.\n\n        The 'rivercolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['rivercolor']

    @rivercolor.setter
    def rivercolor(self, val):
        if False:
            print('Hello World!')
        self['rivercolor'] = val

    @property
    def riverwidth(self):
        if False:
            while True:
                i = 10
        "\n        Sets the stroke width (in px) of the rivers.\n\n        The 'riverwidth' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['riverwidth']

    @riverwidth.setter
    def riverwidth(self, val):
        if False:
            print('Hello World!')
        self['riverwidth'] = val

    @property
    def scope(self):
        if False:
            i = 10
            return i + 15
        "\n        Set the scope of the map.\n\n        The 'scope' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['africa', 'asia', 'europe', 'north america', 'south\n                america', 'usa', 'world']\n\n        Returns\n        -------\n        Any\n        "
        return self['scope']

    @scope.setter
    def scope(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['scope'] = val

    @property
    def showcoastlines(self):
        if False:
            return 10
        "\n        Sets whether or not the coastlines are drawn.\n\n        The 'showcoastlines' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['showcoastlines']

    @showcoastlines.setter
    def showcoastlines(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['showcoastlines'] = val

    @property
    def showcountries(self):
        if False:
            print('Hello World!')
        "\n        Sets whether or not country boundaries are drawn.\n\n        The 'showcountries' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['showcountries']

    @showcountries.setter
    def showcountries(self, val):
        if False:
            while True:
                i = 10
        self['showcountries'] = val

    @property
    def showframe(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets whether or not a frame is drawn around the map.\n\n        The 'showframe' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['showframe']

    @showframe.setter
    def showframe(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['showframe'] = val

    @property
    def showlakes(self):
        if False:
            return 10
        "\n        Sets whether or not lakes are drawn.\n\n        The 'showlakes' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['showlakes']

    @showlakes.setter
    def showlakes(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['showlakes'] = val

    @property
    def showland(self):
        if False:
            print('Hello World!')
        "\n        Sets whether or not land masses are filled in color.\n\n        The 'showland' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['showland']

    @showland.setter
    def showland(self, val):
        if False:
            i = 10
            return i + 15
        self['showland'] = val

    @property
    def showocean(self):
        if False:
            while True:
                i = 10
        "\n        Sets whether or not oceans are filled in color.\n\n        The 'showocean' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['showocean']

    @showocean.setter
    def showocean(self, val):
        if False:
            while True:
                i = 10
        self['showocean'] = val

    @property
    def showrivers(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets whether or not rivers are drawn.\n\n        The 'showrivers' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['showrivers']

    @showrivers.setter
    def showrivers(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['showrivers'] = val

    @property
    def showsubunits(self):
        if False:
            while True:
                i = 10
        "\n        Sets whether or not boundaries of subunits within countries\n        (e.g. states, provinces) are drawn.\n\n        The 'showsubunits' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['showsubunits']

    @showsubunits.setter
    def showsubunits(self, val):
        if False:
            return 10
        self['showsubunits'] = val

    @property
    def subunitcolor(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the color of the subunits boundaries.\n\n        The 'subunitcolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['subunitcolor']

    @subunitcolor.setter
    def subunitcolor(self, val):
        if False:
            print('Hello World!')
        self['subunitcolor'] = val

    @property
    def subunitwidth(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the stroke width (in px) of the subunits boundaries.\n\n        The 'subunitwidth' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['subunitwidth']

    @subunitwidth.setter
    def subunitwidth(self, val):
        if False:
            i = 10
            return i + 15
        self['subunitwidth'] = val

    @property
    def uirevision(self):
        if False:
            while True:
                i = 10
        "\n        Controls persistence of user-driven changes in the view\n        (projection and center). Defaults to `layout.uirevision`.\n\n        The 'uirevision' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['uirevision']

    @uirevision.setter
    def uirevision(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['uirevision'] = val

    @property
    def visible(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the default visibility of the base layers.\n\n        The 'visible' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['visible']

    @visible.setter
    def visible(self, val):
        if False:
            i = 10
            return i + 15
        self['visible'] = val

    @property
    def _prop_descriptions(self):
        if False:
            print('Hello World!')
        return '        bgcolor\n            Set the background color of the map\n        center\n            :class:`plotly.graph_objects.layout.geo.Center`\n            instance or dict with compatible properties\n        coastlinecolor\n            Sets the coastline color.\n        coastlinewidth\n            Sets the coastline stroke width (in px).\n        countrycolor\n            Sets line color of the country boundaries.\n        countrywidth\n            Sets line width (in px) of the country boundaries.\n        domain\n            :class:`plotly.graph_objects.layout.geo.Domain`\n            instance or dict with compatible properties\n        fitbounds\n            Determines if this subplot\'s view settings are auto-\n            computed to fit trace data. On scoped maps, setting\n            `fitbounds` leads to `center.lon` and `center.lat`\n            getting auto-filled. On maps with a non-clipped\n            projection, setting `fitbounds` leads to `center.lon`,\n            `center.lat`, and `projection.rotation.lon` getting\n            auto-filled. On maps with a clipped projection, setting\n            `fitbounds` leads to `center.lon`, `center.lat`,\n            `projection.rotation.lon`, `projection.rotation.lat`,\n            `lonaxis.range` and `lonaxis.range` getting auto-\n            filled. If "locations", only the trace\'s visible\n            locations are considered in the `fitbounds`\n            computations. If "geojson", the entire trace input\n            `geojson` (if provided) is considered in the\n            `fitbounds` computations, Defaults to False.\n        framecolor\n            Sets the color the frame.\n        framewidth\n            Sets the stroke width (in px) of the frame.\n        lakecolor\n            Sets the color of the lakes.\n        landcolor\n            Sets the land mass color.\n        lataxis\n            :class:`plotly.graph_objects.layout.geo.Lataxis`\n            instance or dict with compatible properties\n        lonaxis\n            :class:`plotly.graph_objects.layout.geo.Lonaxis`\n            instance or dict with compatible properties\n        oceancolor\n            Sets the ocean color\n        projection\n            :class:`plotly.graph_objects.layout.geo.Projection`\n            instance or dict with compatible properties\n        resolution\n            Sets the resolution of the base layers. The values have\n            units of km/mm e.g. 110 corresponds to a scale ratio of\n            1:110,000,000.\n        rivercolor\n            Sets color of the rivers.\n        riverwidth\n            Sets the stroke width (in px) of the rivers.\n        scope\n            Set the scope of the map.\n        showcoastlines\n            Sets whether or not the coastlines are drawn.\n        showcountries\n            Sets whether or not country boundaries are drawn.\n        showframe\n            Sets whether or not a frame is drawn around the map.\n        showlakes\n            Sets whether or not lakes are drawn.\n        showland\n            Sets whether or not land masses are filled in color.\n        showocean\n            Sets whether or not oceans are filled in color.\n        showrivers\n            Sets whether or not rivers are drawn.\n        showsubunits\n            Sets whether or not boundaries of subunits within\n            countries (e.g. states, provinces) are drawn.\n        subunitcolor\n            Sets the color of the subunits boundaries.\n        subunitwidth\n            Sets the stroke width (in px) of the subunits\n            boundaries.\n        uirevision\n            Controls persistence of user-driven changes in the view\n            (projection and center). Defaults to\n            `layout.uirevision`.\n        visible\n            Sets the default visibility of the base layers.\n        '

    def __init__(self, arg=None, bgcolor=None, center=None, coastlinecolor=None, coastlinewidth=None, countrycolor=None, countrywidth=None, domain=None, fitbounds=None, framecolor=None, framewidth=None, lakecolor=None, landcolor=None, lataxis=None, lonaxis=None, oceancolor=None, projection=None, resolution=None, rivercolor=None, riverwidth=None, scope=None, showcoastlines=None, showcountries=None, showframe=None, showlakes=None, showland=None, showocean=None, showrivers=None, showsubunits=None, subunitcolor=None, subunitwidth=None, uirevision=None, visible=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Construct a new Geo object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of :class:`plotly.graph_objs.layout.Geo`\n        bgcolor\n            Set the background color of the map\n        center\n            :class:`plotly.graph_objects.layout.geo.Center`\n            instance or dict with compatible properties\n        coastlinecolor\n            Sets the coastline color.\n        coastlinewidth\n            Sets the coastline stroke width (in px).\n        countrycolor\n            Sets line color of the country boundaries.\n        countrywidth\n            Sets line width (in px) of the country boundaries.\n        domain\n            :class:`plotly.graph_objects.layout.geo.Domain`\n            instance or dict with compatible properties\n        fitbounds\n            Determines if this subplot\'s view settings are auto-\n            computed to fit trace data. On scoped maps, setting\n            `fitbounds` leads to `center.lon` and `center.lat`\n            getting auto-filled. On maps with a non-clipped\n            projection, setting `fitbounds` leads to `center.lon`,\n            `center.lat`, and `projection.rotation.lon` getting\n            auto-filled. On maps with a clipped projection, setting\n            `fitbounds` leads to `center.lon`, `center.lat`,\n            `projection.rotation.lon`, `projection.rotation.lat`,\n            `lonaxis.range` and `lonaxis.range` getting auto-\n            filled. If "locations", only the trace\'s visible\n            locations are considered in the `fitbounds`\n            computations. If "geojson", the entire trace input\n            `geojson` (if provided) is considered in the\n            `fitbounds` computations, Defaults to False.\n        framecolor\n            Sets the color the frame.\n        framewidth\n            Sets the stroke width (in px) of the frame.\n        lakecolor\n            Sets the color of the lakes.\n        landcolor\n            Sets the land mass color.\n        lataxis\n            :class:`plotly.graph_objects.layout.geo.Lataxis`\n            instance or dict with compatible properties\n        lonaxis\n            :class:`plotly.graph_objects.layout.geo.Lonaxis`\n            instance or dict with compatible properties\n        oceancolor\n            Sets the ocean color\n        projection\n            :class:`plotly.graph_objects.layout.geo.Projection`\n            instance or dict with compatible properties\n        resolution\n            Sets the resolution of the base layers. The values have\n            units of km/mm e.g. 110 corresponds to a scale ratio of\n            1:110,000,000.\n        rivercolor\n            Sets color of the rivers.\n        riverwidth\n            Sets the stroke width (in px) of the rivers.\n        scope\n            Set the scope of the map.\n        showcoastlines\n            Sets whether or not the coastlines are drawn.\n        showcountries\n            Sets whether or not country boundaries are drawn.\n        showframe\n            Sets whether or not a frame is drawn around the map.\n        showlakes\n            Sets whether or not lakes are drawn.\n        showland\n            Sets whether or not land masses are filled in color.\n        showocean\n            Sets whether or not oceans are filled in color.\n        showrivers\n            Sets whether or not rivers are drawn.\n        showsubunits\n            Sets whether or not boundaries of subunits within\n            countries (e.g. states, provinces) are drawn.\n        subunitcolor\n            Sets the color of the subunits boundaries.\n        subunitwidth\n            Sets the stroke width (in px) of the subunits\n            boundaries.\n        uirevision\n            Controls persistence of user-driven changes in the view\n            (projection and center). Defaults to\n            `layout.uirevision`.\n        visible\n            Sets the default visibility of the base layers.\n\n        Returns\n        -------\n        Geo\n        '
        super(Geo, self).__init__('geo')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.Geo\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.Geo`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('bgcolor', None)
        _v = bgcolor if bgcolor is not None else _v
        if _v is not None:
            self['bgcolor'] = _v
        _v = arg.pop('center', None)
        _v = center if center is not None else _v
        if _v is not None:
            self['center'] = _v
        _v = arg.pop('coastlinecolor', None)
        _v = coastlinecolor if coastlinecolor is not None else _v
        if _v is not None:
            self['coastlinecolor'] = _v
        _v = arg.pop('coastlinewidth', None)
        _v = coastlinewidth if coastlinewidth is not None else _v
        if _v is not None:
            self['coastlinewidth'] = _v
        _v = arg.pop('countrycolor', None)
        _v = countrycolor if countrycolor is not None else _v
        if _v is not None:
            self['countrycolor'] = _v
        _v = arg.pop('countrywidth', None)
        _v = countrywidth if countrywidth is not None else _v
        if _v is not None:
            self['countrywidth'] = _v
        _v = arg.pop('domain', None)
        _v = domain if domain is not None else _v
        if _v is not None:
            self['domain'] = _v
        _v = arg.pop('fitbounds', None)
        _v = fitbounds if fitbounds is not None else _v
        if _v is not None:
            self['fitbounds'] = _v
        _v = arg.pop('framecolor', None)
        _v = framecolor if framecolor is not None else _v
        if _v is not None:
            self['framecolor'] = _v
        _v = arg.pop('framewidth', None)
        _v = framewidth if framewidth is not None else _v
        if _v is not None:
            self['framewidth'] = _v
        _v = arg.pop('lakecolor', None)
        _v = lakecolor if lakecolor is not None else _v
        if _v is not None:
            self['lakecolor'] = _v
        _v = arg.pop('landcolor', None)
        _v = landcolor if landcolor is not None else _v
        if _v is not None:
            self['landcolor'] = _v
        _v = arg.pop('lataxis', None)
        _v = lataxis if lataxis is not None else _v
        if _v is not None:
            self['lataxis'] = _v
        _v = arg.pop('lonaxis', None)
        _v = lonaxis if lonaxis is not None else _v
        if _v is not None:
            self['lonaxis'] = _v
        _v = arg.pop('oceancolor', None)
        _v = oceancolor if oceancolor is not None else _v
        if _v is not None:
            self['oceancolor'] = _v
        _v = arg.pop('projection', None)
        _v = projection if projection is not None else _v
        if _v is not None:
            self['projection'] = _v
        _v = arg.pop('resolution', None)
        _v = resolution if resolution is not None else _v
        if _v is not None:
            self['resolution'] = _v
        _v = arg.pop('rivercolor', None)
        _v = rivercolor if rivercolor is not None else _v
        if _v is not None:
            self['rivercolor'] = _v
        _v = arg.pop('riverwidth', None)
        _v = riverwidth if riverwidth is not None else _v
        if _v is not None:
            self['riverwidth'] = _v
        _v = arg.pop('scope', None)
        _v = scope if scope is not None else _v
        if _v is not None:
            self['scope'] = _v
        _v = arg.pop('showcoastlines', None)
        _v = showcoastlines if showcoastlines is not None else _v
        if _v is not None:
            self['showcoastlines'] = _v
        _v = arg.pop('showcountries', None)
        _v = showcountries if showcountries is not None else _v
        if _v is not None:
            self['showcountries'] = _v
        _v = arg.pop('showframe', None)
        _v = showframe if showframe is not None else _v
        if _v is not None:
            self['showframe'] = _v
        _v = arg.pop('showlakes', None)
        _v = showlakes if showlakes is not None else _v
        if _v is not None:
            self['showlakes'] = _v
        _v = arg.pop('showland', None)
        _v = showland if showland is not None else _v
        if _v is not None:
            self['showland'] = _v
        _v = arg.pop('showocean', None)
        _v = showocean if showocean is not None else _v
        if _v is not None:
            self['showocean'] = _v
        _v = arg.pop('showrivers', None)
        _v = showrivers if showrivers is not None else _v
        if _v is not None:
            self['showrivers'] = _v
        _v = arg.pop('showsubunits', None)
        _v = showsubunits if showsubunits is not None else _v
        if _v is not None:
            self['showsubunits'] = _v
        _v = arg.pop('subunitcolor', None)
        _v = subunitcolor if subunitcolor is not None else _v
        if _v is not None:
            self['subunitcolor'] = _v
        _v = arg.pop('subunitwidth', None)
        _v = subunitwidth if subunitwidth is not None else _v
        if _v is not None:
            self['subunitwidth'] = _v
        _v = arg.pop('uirevision', None)
        _v = uirevision if uirevision is not None else _v
        if _v is not None:
            self['uirevision'] = _v
        _v = arg.pop('visible', None)
        _v = visible if visible is not None else _v
        if _v is not None:
            self['visible'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False