import hashlib
import json
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, cast
from typing_extensions import Final
from streamlit import config
from streamlit.proto.DeckGlJsonChart_pb2 import DeckGlJsonChart as PydeckProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.util import HASHLIB_KWARGS
if TYPE_CHECKING:
    from pydeck import Deck
    from streamlit.delta_generator import DeltaGenerator
EMPTY_MAP: Final[Mapping[str, Any]] = {'initialViewState': {'latitude': 0, 'longitude': 0, 'pitch': 0, 'zoom': 1}}

class PydeckMixin:

    @gather_metrics('pydeck_chart')
    def pydeck_chart(self, pydeck_obj: Optional['Deck']=None, use_container_width: bool=False) -> 'DeltaGenerator':
        if False:
            i = 10
            return i + 15
        "Draw a chart using the PyDeck library.\n\n        This supports 3D maps, point clouds, and more! More info about PyDeck\n        at https://deckgl.readthedocs.io/en/latest/.\n\n        These docs are also quite useful:\n\n        - DeckGL docs: https://github.com/uber/deck.gl/tree/master/docs\n        - DeckGL JSON docs: https://github.com/uber/deck.gl/tree/master/modules/json\n\n        When using this command, Mapbox provides the map tiles to render map\n        content. Note that Mapbox is a third-party product, the use of which is\n        governed by Mapbox's Terms of Use.\n\n        Mapbox requires users to register and provide a token before users can\n        request map tiles. Currently, Streamlit provides this token for you, but\n        this could change at any time. We strongly recommend all users create and\n        use their own personal Mapbox token to avoid any disruptions to their\n        experience. You can do this with the ``mapbox.token`` config option.\n\n        To get a token for yourself, create an account at https://mapbox.com.\n        For more info on how to set config options, see\n        https://docs.streamlit.io/library/advanced-features/configuration\n\n        Parameters\n        ----------\n        pydeck_obj: pydeck.Deck or None\n            Object specifying the PyDeck chart to draw.\n        use_container_width: bool\n\n        Example\n        -------\n        Here's a chart using a HexagonLayer and a ScatterplotLayer. It uses either the\n        light or dark map style, based on which Streamlit theme is currently active:\n\n        >>> import streamlit as st\n        >>> import pandas as pd\n        >>> import numpy as np\n        >>> import pydeck as pdk\n        >>>\n        >>> chart_data = pd.DataFrame(\n        ...    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],\n        ...    columns=['lat', 'lon'])\n        >>>\n        >>> st.pydeck_chart(pdk.Deck(\n        ...     map_style=None,\n        ...     initial_view_state=pdk.ViewState(\n        ...         latitude=37.76,\n        ...         longitude=-122.4,\n        ...         zoom=11,\n        ...         pitch=50,\n        ...     ),\n        ...     layers=[\n        ...         pdk.Layer(\n        ...            'HexagonLayer',\n        ...            data=chart_data,\n        ...            get_position='[lon, lat]',\n        ...            radius=200,\n        ...            elevation_scale=4,\n        ...            elevation_range=[0, 1000],\n        ...            pickable=True,\n        ...            extruded=True,\n        ...         ),\n        ...         pdk.Layer(\n        ...             'ScatterplotLayer',\n        ...             data=chart_data,\n        ...             get_position='[lon, lat]',\n        ...             get_color='[200, 30, 0, 160]',\n        ...             get_radius=200,\n        ...         ),\n        ...     ],\n        ... ))\n\n        .. output::\n           https://doc-pydeck-chart.streamlit.app/\n           height: 530px\n\n        .. note::\n           To make the PyDeck chart's style consistent with Streamlit's theme,\n           you can set ``map_style=None`` in the ``pydeck.Deck`` object.\n\n        "
        pydeck_proto = PydeckProto()
        marshall(pydeck_proto, pydeck_obj, use_container_width)
        return self.dg._enqueue('deck_gl_json_chart', pydeck_proto)

    @property
    def dg(self) -> 'DeltaGenerator':
        if False:
            i = 10
            return i + 15
        'Get our DeltaGenerator.'
        return cast('DeltaGenerator', self)

def _get_pydeck_tooltip(pydeck_obj: Optional['Deck']) -> Optional[Dict[str, str]]:
    if False:
        print('Hello World!')
    if pydeck_obj is None:
        return None
    desk_widget = getattr(pydeck_obj, 'deck_widget', None)
    if desk_widget is not None and isinstance(desk_widget.tooltip, dict):
        return desk_widget.tooltip
    tooltip = getattr(pydeck_obj, '_tooltip', None)
    if tooltip is not None and isinstance(tooltip, dict):
        return tooltip
    return None

def marshall(pydeck_proto: PydeckProto, pydeck_obj: Optional['Deck'], use_container_width: bool) -> None:
    if False:
        while True:
            i = 10
    if pydeck_obj is None:
        spec = json.dumps(EMPTY_MAP)
        id = ''
    else:
        spec = pydeck_obj.to_json()
        json_string = json.dumps(spec)
        json_bytes = json_string.encode('utf-8')
        id = hashlib.md5(json_bytes, **HASHLIB_KWARGS).hexdigest()
    pydeck_proto.json = spec
    pydeck_proto.use_container_width = use_container_width
    pydeck_proto.id = id
    tooltip = _get_pydeck_tooltip(pydeck_obj)
    if tooltip:
        pydeck_proto.tooltip = json.dumps(tooltip)
    mapbox_token = config.get_option('mapbox.token')
    if mapbox_token:
        pydeck_proto.mapbox_token = mapbox_token