from typing import TYPE_CHECKING, cast
from streamlit.proto.Balloons_pb2 import Balloons as BalloonsProto
from streamlit.runtime.metrics_util import gather_metrics
if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

class BalloonsMixin:

    @gather_metrics('balloons')
    def balloons(self) -> 'DeltaGenerator':
        if False:
            i = 10
            return i + 15
        'Draw celebratory balloons.\n\n        Example\n        -------\n        >>> import streamlit as st\n        >>>\n        >>> st.balloons()\n\n        ...then watch your app and get ready for a celebration!\n\n        '
        balloons_proto = BalloonsProto()
        balloons_proto.show = True
        return self.dg._enqueue('balloons', balloons_proto)

    @property
    def dg(self) -> 'DeltaGenerator':
        if False:
            while True:
                i = 10
        'Get our DeltaGenerator.'
        return cast('DeltaGenerator', self)