from typing import TYPE_CHECKING, Optional, cast
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Toast_pb2 import Toast as ToastProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.string_util import clean_text, validate_emoji
from streamlit.type_util import SupportsStr
if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

def validate_text(toast_text: SupportsStr) -> SupportsStr:
    if False:
        while True:
            i = 10
    if str(toast_text) == '':
        raise StreamlitAPIException(f'Toast body cannot be blank - please provide a message.')
    else:
        return toast_text

class ToastMixin:

    @gather_metrics('toast')
    def toast(self, body: SupportsStr, *, icon: Optional[str]=None) -> 'DeltaGenerator':
        if False:
            while True:
                i = 10
        'Display a short message, known as a notification "toast".\n        The toast appears in the app\'s bottom-right corner and disappears after four seconds.\n\n        .. warning::\n            ``st.toast`` is not compatible with Streamlit\'s `caching             <https://docs.streamlit.io/library/advanced-features/caching>`_ and\n            cannot be called within a cached function.\n\n        Parameters\n        ----------\n        body : str\n            The string to display as Github-flavored Markdown. Syntax\n            information can be found at: https://github.github.com/gfm.\n\n            This also supports:\n\n            * Emoji shortcodes, such as ``:+1:``  and ``:sunglasses:``.\n              For a list of all supported codes,\n              see https://share.streamlit.io/streamlit/emoji-shortcodes.\n\n            * LaTeX expressions, by wrapping them in "$" or "$$" (the "$$"\n              must be on their own lines). Supported LaTeX functions are listed\n              at https://katex.org/docs/supported.html.\n\n            * Colored text, using the syntax ``:color[text to be colored]``,\n              where ``color`` needs to be replaced with any of the following\n              supported colors: blue, green, orange, red, violet, gray/grey, rainbow.\n        icon : str or None\n            An optional argument that specifies an emoji to use as\n            the icon for the toast. Shortcodes are not allowed, please use a\n            single character instead. E.g. "🚨", "🔥", "🤖", etc.\n            Defaults to None, which means no icon is displayed.\n\n        Example\n        -------\n        >>> import streamlit as st\n        >>>\n        >>> st.toast(\'Your edited image was saved!\', icon=\'😍\')\n        '
        toast_proto = ToastProto()
        toast_proto.body = clean_text(validate_text(body))
        toast_proto.icon = validate_emoji(icon)
        return self.dg._enqueue('toast', toast_proto)

    @property
    def dg(self) -> 'DeltaGenerator':
        if False:
            while True:
                i = 10
        'Get our DeltaGenerator.'
        return cast('DeltaGenerator', self)