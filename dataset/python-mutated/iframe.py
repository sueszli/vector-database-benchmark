from typing import TYPE_CHECKING, Optional, cast
from streamlit.proto.IFrame_pb2 import IFrame as IFrameProto
from streamlit.runtime.metrics_util import gather_metrics
if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

class IframeMixin:

    @gather_metrics('_iframe')
    def _iframe(self, src: str, width: Optional[int]=None, height: Optional[int]=None, scrolling: bool=False) -> 'DeltaGenerator':
        if False:
            return 10
        "Load a remote URL in an iframe.\n\n        Parameters\n        ----------\n        src : str\n            The URL of the page to embed.\n        width : int\n            The width of the frame in CSS pixels. Defaults to the app's\n            default element width.\n        height : int\n            The height of the frame in CSS pixels. Defaults to 150.\n        scrolling : bool\n            If True, show a scrollbar when the content is larger than the iframe.\n            Otherwise, do not show a scrollbar. Defaults to False.\n\n        "
        iframe_proto = IFrameProto()
        marshall(iframe_proto, src=src, width=width, height=height, scrolling=scrolling)
        return self.dg._enqueue('iframe', iframe_proto)

    @gather_metrics('_html')
    def _html(self, html: str, width: Optional[int]=None, height: Optional[int]=None, scrolling: bool=False) -> 'DeltaGenerator':
        if False:
            for i in range(10):
                print('nop')
        "Display an HTML string in an iframe.\n\n        Parameters\n        ----------\n        html : str\n            The HTML string to embed in the iframe.\n        width : int\n            The width of the frame in CSS pixels. Defaults to the app's\n            default element width.\n        height : int\n            The height of the frame in CSS pixels. Defaults to 150.\n        scrolling : bool\n            If True, show a scrollbar when the content is larger than the iframe.\n            Otherwise, do not show a scrollbar. Defaults to False.\n\n        "
        iframe_proto = IFrameProto()
        marshall(iframe_proto, srcdoc=html, width=width, height=height, scrolling=scrolling)
        return self.dg._enqueue('iframe', iframe_proto)

    @property
    def dg(self) -> 'DeltaGenerator':
        if False:
            while True:
                i = 10
        'Get our DeltaGenerator.'
        return cast('DeltaGenerator', self)

def marshall(proto: IFrameProto, src: Optional[str]=None, srcdoc: Optional[str]=None, width: Optional[int]=None, height: Optional[int]=None, scrolling: bool=False) -> None:
    if False:
        for i in range(10):
            print('nop')
    "Marshalls data into an IFrame proto.\n\n    These parameters correspond directly to <iframe> attributes, which are\n    described in more detail at\n    https://developer.mozilla.org/en-US/docs/Web/HTML/Element/iframe.\n\n    Parameters\n    ----------\n    proto : IFrame protobuf\n        The protobuf object to marshall data into.\n    src : str\n        The URL of the page to embed.\n    srcdoc : str\n        Inline HTML to embed. Overrides src.\n    width : int\n        The width of the frame in CSS pixels. Defaults to the app's\n        default element width.\n    height : int\n        The height of the frame in CSS pixels. Defaults to 150.\n    scrolling : bool\n        If true, show a scrollbar when the content is larger than the iframe.\n        Otherwise, never show a scrollbar.\n\n    "
    if src is not None:
        proto.src = src
    if srcdoc is not None:
        proto.srcdoc = srcdoc
    if width is not None:
        proto.width = width
        proto.has_width = True
    if height is not None:
        proto.height = height
    else:
        proto.height = 150
    proto.scrolling = scrolling