"""
This example demonstrates a code editor widget based on CodeMirror.
"""
from flexx import flx
base_url = 'https://cdnjs.cloudflare.com/ajax/libs/codemirror/'
flx.assets.associate_asset(__name__, base_url + '5.21.0/codemirror.min.css')
flx.assets.associate_asset(__name__, base_url + '5.21.0/codemirror.min.js')
flx.assets.associate_asset(__name__, base_url + '5.21.0/mode/python/python.js')
flx.assets.associate_asset(__name__, base_url + '5.21.0/theme/solarized.css')
flx.assets.associate_asset(__name__, base_url + '5.21.0/addon/selection/active-line.js')
flx.assets.associate_asset(__name__, base_url + '5.21.0/addon/edit/matchbrackets.js')

class CodeEditor(flx.Widget):
    """ A CodeEditor widget based on CodeMirror.
    """
    CSS = '\n    .flx-CodeEditor > .CodeMirror {\n        width: 100%;\n        height: 100%;\n    }\n    '

    def init(self):
        if False:
            for i in range(10):
                print('nop')
        global window
        options = dict(value='import os\n\ndirs = os.walk', mode='python', theme='solarized dark', autofocus=True, styleActiveLine=True, matchBrackets=True, indentUnit=4, smartIndent=True, lineWrapping=True, lineNumbers=True, firstLineNumber=1, readOnly=False)
        self.cm = window.CodeMirror(self.node, options)

    @flx.reaction('size')
    def __on_size(self, *events):
        if False:
            for i in range(10):
                print('nop')
        self.cm.refresh()
if __name__ == '__main__':
    flx.launch(CodeEditor, 'app')
    flx.run()