from typing import TYPE_CHECKING, cast
from streamlit.proto.Snow_pb2 import Snow as SnowProto
from streamlit.runtime.metrics_util import gather_metrics
if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

class SnowMixin:

    @gather_metrics('snow')
    def snow(self) -> 'DeltaGenerator':
        if False:
            while True:
                i = 10
        'Draw celebratory snowfall.\n\n        Example\n        -------\n        >>> import streamlit as st\n        >>>\n        >>> st.snow()\n\n        ...then watch your app and get ready for a cool celebration!\n\n        '
        snow_proto = SnowProto()
        snow_proto.show = True
        return self.dg._enqueue('snow', snow_proto)

    @property
    def dg(self) -> 'DeltaGenerator':
        if False:
            i = 10
            return i + 15
        'Get our DeltaGenerator.'
        return cast('DeltaGenerator', self)