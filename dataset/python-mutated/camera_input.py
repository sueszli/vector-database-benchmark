from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, Optional, Union, cast
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import check_callback_rules, check_session_state_rules, get_label_visibility_proto_value
from streamlit.elements.widgets.file_uploader import _get_upload_files
from streamlit.proto.CameraInput_pb2 import CameraInput as CameraInputProto
from streamlit.proto.Common_pb2 import FileUploaderState as FileUploaderStateProto
from streamlit.proto.Common_pb2 import UploadedFileInfo as UploadedFileInfoProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import WidgetArgs, WidgetCallback, WidgetKwargs, register_widget
from streamlit.runtime.state.common import compute_widget_id
from streamlit.runtime.uploaded_file_manager import DeletedFile, UploadedFile
from streamlit.type_util import Key, LabelVisibility, maybe_raise_label_warnings, to_key
if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator
SomeUploadedSnapshotFile = Union[UploadedFile, DeletedFile, None]

@dataclass
class CameraInputSerde:

    def serialize(self, snapshot: SomeUploadedSnapshotFile) -> FileUploaderStateProto:
        if False:
            print('Hello World!')
        state_proto = FileUploaderStateProto()
        if snapshot is None or isinstance(snapshot, DeletedFile):
            return state_proto
        file_info: UploadedFileInfoProto = state_proto.uploaded_file_info.add()
        file_info.file_id = snapshot.file_id
        file_info.name = snapshot.name
        file_info.size = snapshot.size
        file_info.file_urls.CopyFrom(snapshot._file_urls)
        return state_proto

    def deserialize(self, ui_value: Optional[FileUploaderStateProto], widget_id: str) -> SomeUploadedSnapshotFile:
        if False:
            return 10
        upload_files = _get_upload_files(ui_value)
        if len(upload_files) == 0:
            return_value = None
        else:
            return_value = upload_files[0]
        return return_value

class CameraInputMixin:

    @gather_metrics('camera_input')
    def camera_input(self, label: str, key: Optional[Key]=None, help: Optional[str]=None, on_change: Optional[WidgetCallback]=None, args: Optional[WidgetArgs]=None, kwargs: Optional[WidgetKwargs]=None, *, disabled: bool=False, label_visibility: LabelVisibility='visible') -> Optional[UploadedFile]:
        if False:
            print('Hello World!')
        'Display a widget that returns pictures from the user\'s webcam.\n\n        Parameters\n        ----------\n        label : str\n            A short label explaining to the user what this widget is used for.\n            The label can optionally contain Markdown and supports the following\n            elements: Bold, Italics, Strikethroughs, Inline Code, Emojis, and Links.\n\n            This also supports:\n\n            * Emoji shortcodes, such as ``:+1:``  and ``:sunglasses:``.\n              For a list of all supported codes,\n              see https://share.streamlit.io/streamlit/emoji-shortcodes.\n\n            * LaTeX expressions, by wrapping them in "$" or "$$" (the "$$"\n              must be on their own lines). Supported LaTeX functions are listed\n              at https://katex.org/docs/supported.html.\n\n            * Colored text, using the syntax ``:color[text to be colored]``,\n              where ``color`` needs to be replaced with any of the following\n              supported colors: blue, green, orange, red, violet, gray/grey, rainbow.\n\n            Unsupported elements are unwrapped so only their children (text contents) render.\n            Display unsupported elements as literal characters by\n            backslash-escaping them. E.g. ``1\\. Not an ordered list``.\n\n            For accessibility reasons, you should never set an empty label (label="")\n            but hide it with label_visibility if needed. In the future, we may disallow\n            empty labels by raising an exception.\n\n        key : str or int\n            An optional string or integer to use as the unique key for the widget.\n            If this is omitted, a key will be generated for the widget\n            based on its content. Multiple widgets of the same type may\n            not share the same key.\n\n        help : str\n            A tooltip that gets displayed next to the camera input.\n\n        on_change : callable\n            An optional callback invoked when this camera_input\'s value\n            changes.\n\n        args : tuple\n            An optional tuple of args to pass to the callback.\n\n        kwargs : dict\n            An optional dict of kwargs to pass to the callback.\n\n        disabled : bool\n            An optional boolean, which disables the camera input if set to\n            True. Default is False.\n        label_visibility : "visible", "hidden", or "collapsed"\n            The visibility of the label. If "hidden", the label doesn\'t show but there\n            is still empty space for it above the widget (equivalent to label="").\n            If "collapsed", both the label and the space are removed. Default is\n            "visible".\n\n        Returns\n        -------\n        None or UploadedFile\n            The UploadedFile class is a subclass of BytesIO, and therefore\n            it is "file-like". This means you can pass them anywhere where\n            a file is expected.\n\n        Examples\n        --------\n        >>> import streamlit as st\n        >>>\n        >>> picture = st.camera_input("Take a picture")\n        >>>\n        >>> if picture:\n        ...     st.image(picture)\n\n        '
        ctx = get_script_run_ctx()
        return self._camera_input(label=label, key=key, help=help, on_change=on_change, args=args, kwargs=kwargs, disabled=disabled, label_visibility=label_visibility, ctx=ctx)

    def _camera_input(self, label: str, key: Optional[Key]=None, help: Optional[str]=None, on_change: Optional[WidgetCallback]=None, args: Optional[WidgetArgs]=None, kwargs: Optional[WidgetKwargs]=None, *, disabled: bool=False, label_visibility: LabelVisibility='visible', ctx: Optional[ScriptRunContext]=None) -> Optional[UploadedFile]:
        if False:
            i = 10
            return i + 15
        key = to_key(key)
        check_callback_rules(self.dg, on_change)
        check_session_state_rules(default_value=None, key=key, writes_allowed=False)
        maybe_raise_label_warnings(label, label_visibility)
        id = compute_widget_id('camera_input', user_key=key, label=label, key=key, help=help, form_id=current_form_id(self.dg), page=ctx.page_script_hash if ctx else None)
        camera_input_proto = CameraInputProto()
        camera_input_proto.id = id
        camera_input_proto.label = label
        camera_input_proto.form_id = current_form_id(self.dg)
        camera_input_proto.disabled = disabled
        camera_input_proto.label_visibility.value = get_label_visibility_proto_value(label_visibility)
        if help is not None:
            camera_input_proto.help = dedent(help)
        serde = CameraInputSerde()
        camera_input_state = register_widget('camera_input', camera_input_proto, user_key=key, on_change_handler=on_change, args=args, kwargs=kwargs, deserializer=serde.deserialize, serializer=serde.serialize, ctx=ctx)
        self.dg._enqueue('camera_input', camera_input_proto)
        if isinstance(camera_input_state.value, DeletedFile):
            return None
        return camera_input_state.value

    @property
    def dg(self) -> 'DeltaGenerator':
        if False:
            i = 10
            return i + 15
        'Get our DeltaGenerator.'
        return cast('DeltaGenerator', self)