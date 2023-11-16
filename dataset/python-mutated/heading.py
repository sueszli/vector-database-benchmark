from enum import Enum
from typing import TYPE_CHECKING, Optional, Union, cast
from typing_extensions import Literal
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Heading_pb2 import Heading as HeadingProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.string_util import clean_text
from streamlit.type_util import SupportsStr
if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

class HeadingProtoTag(Enum):
    TITLE_TAG = 'h1'
    HEADER_TAG = 'h2'
    SUBHEADER_TAG = 'h3'
Anchor = Optional[Union[str, Literal[False]]]
Divider = Optional[Union[bool, str]]

class HeadingMixin:

    @gather_metrics('header')
    def header(self, body: SupportsStr, anchor: Anchor=None, *, help: Optional[str]=None, divider: Divider=False) -> 'DeltaGenerator':
        if False:
            i = 10
            return i + 15
        'Display text in header formatting.\n\n        Parameters\n        ----------\n        body : str\n            The text to display as Github-flavored Markdown. Syntax\n            information can be found at: https://github.github.com/gfm.\n\n            This also supports:\n\n            * Emoji shortcodes, such as ``:+1:``  and ``:sunglasses:``.\n              For a list of all supported codes,\n              see https://share.streamlit.io/streamlit/emoji-shortcodes.\n\n            * LaTeX expressions, by wrapping them in "$" or "$$" (the "$$"\n              must be on their own lines). Supported LaTeX functions are listed\n              at https://katex.org/docs/supported.html.\n\n            * Colored text, using the syntax ``:color[text to be colored]``,\n              where ``color`` needs to be replaced with any of the following\n              supported colors: blue, green, orange, red, violet, gray/grey, rainbow.\n\n        anchor : str or False\n            The anchor name of the header that can be accessed with #anchor\n            in the URL. If omitted, it generates an anchor using the body.\n            If False, the anchor is not shown in the UI.\n\n        help : str\n            An optional tooltip that gets displayed next to the header.\n\n        divider : bool or “blue”, “green”, “orange”, “red”, “violet”, “gray”/"grey", or “rainbow”\n            Shows a colored divider below the header. If True, successive\n            headers will cycle through divider colors. That is, the first\n            header will have a blue line, the second header will have a\n            green line, and so on. If a string, the color can be set to one of\n            the following: blue, green, orange, red, violet, gray/grey, or\n            rainbow.\n\n        Examples\n        --------\n        >>> import streamlit as st\n        >>>\n        >>> st.header(\'This is a header with a divider\', divider=\'rainbow\')\n        >>> st.header(\'_Streamlit_ is :blue[cool] :sunglasses:\')\n\n        .. output::\n           https://doc-header.streamlit.app/\n           height: 220px\n\n        '
        return self.dg._enqueue('heading', HeadingMixin._create_heading_proto(tag=HeadingProtoTag.HEADER_TAG, body=body, anchor=anchor, help=help, divider=divider))

    @gather_metrics('subheader')
    def subheader(self, body: SupportsStr, anchor: Anchor=None, *, help: Optional[str]=None, divider: Divider=False) -> 'DeltaGenerator':
        if False:
            return 10
        'Display text in subheader formatting.\n\n        Parameters\n        ----------\n        body : str\n            The text to display as Github-flavored Markdown. Syntax\n            information can be found at: https://github.github.com/gfm.\n\n            This also supports:\n\n            * Emoji shortcodes, such as ``:+1:``  and ``:sunglasses:``.\n              For a list of all supported codes,\n              see https://share.streamlit.io/streamlit/emoji-shortcodes.\n\n            * LaTeX expressions, by wrapping them in "$" or "$$" (the "$$"\n              must be on their own lines). Supported LaTeX functions are listed\n              at https://katex.org/docs/supported.html.\n\n            * Colored text, using the syntax ``:color[text to be colored]``,\n              where ``color`` needs to be replaced with any of the following\n              supported colors: blue, green, orange, red, violet, gray/grey, rainbow.\n\n        anchor : str or False\n            The anchor name of the header that can be accessed with #anchor\n            in the URL. If omitted, it generates an anchor using the body.\n            If False, the anchor is not shown in the UI.\n\n        help : str\n            An optional tooltip that gets displayed next to the subheader.\n\n        divider : bool or “blue”, “green”, “orange”, “red”, “violet”, “gray”/"grey", or “rainbow”\n            Shows a colored divider below the header. If True, successive\n            headers will cycle through divider colors. That is, the first\n            header will have a blue line, the second header will have a\n            green line, and so on. If a string, the color can be set to one of\n            the following: blue, green, orange, red, violet, gray/grey, or\n            rainbow.\n\n        Examples\n        --------\n        >>> import streamlit as st\n        >>>\n        >>> st.subheader(\'This is a subheader with a divider\', divider=\'rainbow\')\n        >>> st.subheader(\'_Streamlit_ is :blue[cool] :sunglasses:\')\n\n        .. output::\n           https://doc-subheader.streamlit.app/\n           height: 220px\n\n        '
        return self.dg._enqueue('heading', HeadingMixin._create_heading_proto(tag=HeadingProtoTag.SUBHEADER_TAG, body=body, anchor=anchor, help=help, divider=divider))

    @gather_metrics('title')
    def title(self, body: SupportsStr, anchor: Anchor=None, *, help: Optional[str]=None) -> 'DeltaGenerator':
        if False:
            while True:
                i = 10
        'Display text in title formatting.\n\n        Each document should have a single `st.title()`, although this is not\n        enforced.\n\n        Parameters\n        ----------\n        body : str\n            The text to display as Github-flavored Markdown. Syntax\n            information can be found at: https://github.github.com/gfm.\n\n            This also supports:\n\n            * Emoji shortcodes, such as ``:+1:``  and ``:sunglasses:``.\n              For a list of all supported codes,\n              see https://share.streamlit.io/streamlit/emoji-shortcodes.\n\n            * LaTeX expressions, by wrapping them in "$" or "$$" (the "$$"\n              must be on their own lines). Supported LaTeX functions are listed\n              at https://katex.org/docs/supported.html.\n\n            * Colored text, using the syntax ``:color[text to be colored]``,\n              where ``color`` needs to be replaced with any of the following\n              supported colors: blue, green, orange, red, violet, gray/grey, rainbow.\n\n        anchor : str or False\n            The anchor name of the header that can be accessed with #anchor\n            in the URL. If omitted, it generates an anchor using the body.\n            If False, the anchor is not shown in the UI.\n\n        help : str\n            An optional tooltip that gets displayed next to the title.\n\n        Examples\n        --------\n        >>> import streamlit as st\n        >>>\n        >>> st.title(\'This is a title\')\n        >>> st.title(\'_Streamlit_ is :blue[cool] :sunglasses:\')\n\n        .. output::\n           https://doc-title.streamlit.app/\n           height: 220px\n\n        '
        return self.dg._enqueue('heading', HeadingMixin._create_heading_proto(tag=HeadingProtoTag.TITLE_TAG, body=body, anchor=anchor, help=help))

    @property
    def dg(self) -> 'DeltaGenerator':
        if False:
            print('Hello World!')
        'Get our DeltaGenerator.'
        return cast('DeltaGenerator', self)

    @staticmethod
    def _handle_divider_color(divider):
        if False:
            while True:
                i = 10
        if divider is True:
            return 'auto'
        valid_colors = ['blue', 'green', 'orange', 'red', 'violet', 'gray', 'grey', 'rainbow']
        if divider in valid_colors:
            return divider
        else:
            raise StreamlitAPIException(f"Divider parameter has invalid value: `{divider}`. Please choose from: {', '.join(valid_colors)}.")

    @staticmethod
    def _create_heading_proto(tag: HeadingProtoTag, body: SupportsStr, anchor: Anchor=None, help: Optional[str]=None, divider: Divider=False) -> HeadingProto:
        if False:
            while True:
                i = 10
        proto = HeadingProto()
        proto.tag = tag.value
        proto.body = clean_text(body)
        if divider:
            proto.divider = HeadingMixin._handle_divider_color(divider)
        if anchor is not None:
            if anchor is False:
                proto.hide_anchor = True
            elif isinstance(anchor, str):
                proto.anchor = anchor
            elif anchor is True:
                raise StreamlitAPIException('Anchor parameter has invalid value: %s. Supported values: None, any string or False' % anchor)
            else:
                raise StreamlitAPIException('Anchor parameter has invalid type: %s. Supported values: None, any string or False' % type(anchor).__name__)
        if help:
            proto.help = help
        return proto