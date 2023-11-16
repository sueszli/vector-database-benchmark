import uuid
import json
from IPython.display import display, Markdown, HTML
from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import TerminalFormatter

class RenderJSON(object):

    def __init__(self, data):
        if False:
            print('Hello World!')
        self.render_json(data)

    def render_json(self, data):
        if False:
            return 10
        json_object = json.loads(json.dumps(data))
        json_str = json.dumps(json_object, indent=4, sort_keys=True)
        print(highlight(json_str, JsonLexer(), TerminalFormatter()))

class RenderMarkdown(object):

    def __init__(self, markdown_data):
        if False:
            while True:
                i = 10
        self.markdown_data = markdown_data
        self.uuid = str(uuid.uuid4())

    def _ipython_display_(self):
        if False:
            i = 10
            return i + 15
        display(Markdown(self.markdown_data))

class RenderHTML(object):

    def __init__(self, html_data=None, html_file=None):
        if False:
            while True:
                i = 10
        if not html_data and (not html_file):
            print('You need to provide either a filename or raw HTML data for something to be rendered')
        self.html_data = html_data
        self.html_file = html_file
        self.uuid = str(uuid.uuid4())

    def _ipython_display_(self):
        if False:
            print('Hello World!')
        if self.html_data is not None:
            display(HTML(self.html_data))
        if self.html_file is not None:
            HTML(filename=self.filename)