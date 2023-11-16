"""%tensorboard line magic that patches TensorBoard's implementation to make use of Jupyter
TensorBoard server extension providing built-in proxying.
Use:
    %load_ext tensorboard
    %tensorboard --logdir /logs
"""
import argparse
import uuid
from IPython.display import HTML, display

def _tensorboard_magic(line):
    if False:
        i = 10
        return i + 15
    'Line magic function.\n\n    Makes an AJAX call to the Jupyter TensorBoard server extension and outputs\n    an IFrame displaying the TensorBoard instance.\n    '
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', default='/workspace/')
    args = parser.parse_args(line.split())
    iframe_id = 'tensorboard-' + str(uuid.uuid4())
    html = '\n<!-- JUPYTER_TENSORBOARD_TEST_MARKER -->\n<script>\n    fetch(Jupyter.notebook.base_url + \'api/tensorboard\', {\n        method: \'POST\',\n        contentType: \'application/json\',\n        body: JSON.stringify({ \'logdir\': \'%s\' }),\n        headers: { \'Content-Type\': \'application/json\' }\n    })\n        .then(res => res.json())\n        .then(res => {\n            const iframe = document.getElementById(\'%s\');\n            iframe.src = Jupyter.notebook.base_url + \'tensorboard/\' + res.name;\n            iframe.style.display = \'block\';\n        });\n</script>\n<iframe\n    id="%s"\n    style="width: 100%%; height: 620px; display: none;"\n    frameBorder="0">\n</iframe>\n' % (args.logdir, iframe_id, iframe_id)
    display(HTML(html))

def load_ipython_extension(ipython):
    if False:
        for i in range(10):
            print('nop')
    'Deprecated: use `%load_ext tensorboard` instead.\n\n    Raises:\n      RuntimeError: Always.\n    '
    raise RuntimeError("Use '%load_ext tensorboard' instead of '%load_ext tensorboard.notebook'.")

def _load_ipython_extension(ipython):
    if False:
        while True:
            i = 10
    'Load the TensorBoard notebook extension.\n\n    Intended to be called from `%load_ext tensorboard`. Do not invoke this\n    directly.\n\n    Args:\n      ipython: An `IPython.InteractiveShell` instance.\n    '
    ipython.register_magic_function(_tensorboard_magic, magic_kind='line', magic_name='tensorboard')