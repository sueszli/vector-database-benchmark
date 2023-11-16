"""
Paint the specified XObject. The operand name shall appear as a key in
the XObject subdictionary of the current resource dictionary (see 7.8.3,
"Resource Dictionaries"). The associated value shall be a stream whose
Type entry, if present, is XObject.
"""
import io
import typing
import PIL
from borb.io.read.types import AnyPDFType
from borb.io.read.types import Dictionary
from borb.io.read.types import Name
from borb.io.read.types import Stream
from borb.pdf.canvas.event.image_render_event import ImageRenderEvent
from borb.pdf.canvas.operator.canvas_operator import CanvasOperator

class Do(CanvasOperator):
    """
    Paint the specified XObject. The operand name shall appear as a key in
    the XObject subdictionary of the current resource dictionary (see 7.8.3,
    "Resource Dictionaries"). The associated value shall be a stream whose
    Type entry, if present, is XObject.

    The effect of Do depends on the value
    of the XObject’s Subtype entry, which may be Image (see 8.9.5, "Image
    Dictionaries"), Form (see 8.10, "Form XObjects"), or PS (see 8.8.2,
    "PostScript XObjects").
    """

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__('Do', 1)

    def invoke(self, canvas_stream_processor: 'CanvasStreamProcessor', operands: typing.List[AnyPDFType]=[], event_listeners: typing.List['EventListener']=[]) -> None:
        if False:
            return 10
        '\n        Invoke the Do operator\n        '
        canvas = canvas_stream_processor.get_canvas()
        page = canvas_stream_processor.get_page()
        assert isinstance(operands[0], Name)
        xobject = canvas_stream_processor.get_resource('XObject', str(operands[0]))
        if isinstance(xobject, PIL.Image.Image):
            for l in event_listeners:
                l._event_occurred(ImageRenderEvent(graphics_state=canvas.graphics_state, image=xobject))
            return
        if isinstance(xobject, Stream) and 'Subtype' in xobject and (xobject['Subtype'] == 'Form'):
            xobject_resources: Dictionary = xobject['Resources'] if 'Resources' in xobject else {}
            child_canvas_stream_processor = canvas_stream_processor.create_child_canvas_stream_processor([xobject_resources])
            child_canvas_stream_processor.read(io.BytesIO(xobject['DecodedBytes']), event_listeners)
            return
        pass