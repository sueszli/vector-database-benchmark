""" Markdown widget

Widget containing a string which content gets rendered and shown as markdown text.

See the working example from `flexxamples/ui_usage/markdown.py`.

Simple usage:

.. UIExample:: 200

    def init(self):
        content = "# Welcome

"             "Hello.  Welcome to my **website**."             "This is an example of a widget container for markdown content. "             "The content can be text or a link.

"
        ui.Markdown(content=content, style='background:#EAECFF;height:60%;')

"""
from ... import app, event
from . import Widget
app.assets.associate_asset(__name__, 'https://cdnjs.cloudflare.com/ajax/libs/showdown/1.9.1/showdown.min.js')

class Markdown(Widget):
    """ A widget that shows a rendered Markdown content.
    """
    CSS = '\n\n    .flx-Markdown {\n        height: min(100vh,100%);\n        overflow-y: auto;\n    }\n    '
    content = event.StringProp(settable=True, doc='\n        The markdown content to be rendered\n        ')

    @event.reaction
    def __content_change(self):
        if False:
            for i in range(10):
                print('nop')
        global showdown
        conv = showdown.Converter()
        self.node.innerHTML = conv.makeHtml(self.content)