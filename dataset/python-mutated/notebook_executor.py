"""Module to execute jupyter notebooks and gather the output into renderable
HTML files."""
import html
import os
import shutil
import subprocess
from html.parser import HTMLParser
from apache_beam.runners.interactive.utils import obfuscate
try:
    import nbformat
    from jupyter_client.kernelspec import KernelSpecManager
    from nbconvert.preprocessors import ExecutePreprocessor
    _interactive_integration_ready = True
except ImportError:
    _interactive_integration_ready = False

class NotebookExecutor(object):
    """Executor that reads notebooks, executes it and gathers outputs into static
  HTML pages that can be served."""

    def __init__(self, path):
        if False:
            print('Hello World!')
        assert _interactive_integration_ready, '[interactive_test] dependency is not installed.'
        assert os.path.exists(path), '{} does not exist.'.format(path)
        self._paths = []
        if os.path.isdir(path):
            for (root, _, files) in os.walk(path):
                for filename in files:
                    if filename.endswith('.ipynb'):
                        self._paths.append(os.path.join(root, filename))
        elif path.endswith('.ipynb'):
            self._paths.append(path)
        assert len(self._paths) > 0, 'No notebooks to be executed under{}'.format(path)
        self._dir = os.path.dirname(self._paths[0])
        self._output_html_dir = os.path.join(self._dir, 'output')
        self.cleanup()
        self._output_html_paths = {}
        self._notebook_path_to_execution_id = {}
        kernel_specs = KernelSpecManager().get_all_specs()
        if 'test' not in kernel_specs:
            process = subprocess.run(['python', '-m', 'ipykernel', 'install', '--user', '--name', 'test'], check=True)
            process.check_returncode()

    def cleanup(self):
        if False:
            while True:
                i = 10
        'Cleans up the output folder.'
        _cleanup(self._output_html_dir)

    def execute(self):
        if False:
            for i in range(10):
                print('nop')
        'Executes all notebooks found in the scoped path and gathers their\n    outputs into HTML pages stored in the output folder.'
        for path in self._paths:
            with open(path, 'r') as nb_f:
                nb = nbformat.read(nb_f, as_version=4)
                ep = ExecutePreprocessor(timeout=-1, allow_errors=True, kernel_name='test')
                ep.preprocess(nb, {'metadata': {'path': os.path.dirname(path)}})
            execution_id = obfuscate(path)
            output_html_path = os.path.join(self._output_html_dir, execution_id + '.html')
            with open(output_html_path, 'a+') as sink:
                sink.write('<html>\n')
                sink.write('<head>\n')
                sink.write('</head>\n')
                sink.write('<body>\n')
                for cell in nb['cells']:
                    if cell['cell_type'] == 'code':
                        for output in cell['outputs']:
                            _extract_html(output, sink)
                sink.write('</body>\n')
                sink.write('</html>\n')
            self._output_html_paths[execution_id] = output_html_path
            self._notebook_path_to_execution_id[path] = execution_id

    @property
    def output_html_paths(self):
        if False:
            i = 10
            return i + 15
        'Mapping from execution ids to output html page paths.\n\n    An execution/test id is an obfuscated value from the executed notebook path.\n    It identifies the input notebook, the output html, the screenshot of the\n    output html, and the golden screenshot for comparison.\n    '
        return self._output_html_paths

    @property
    def output_html_dir(self):
        if False:
            print('Hello World!')
        "The directory's path to all the output html pages generated."
        return self._output_html_dir

    @property
    def notebook_path_to_execution_id(self):
        if False:
            while True:
                i = 10
        'Mapping from input notebook paths to their obfuscated execution ids.'
        return self._notebook_path_to_execution_id

def _cleanup(output_dir):
    if False:
        i = 10
        return i + 15
    'Cleans up the given output_dir.'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

def _extract_html(output, sink):
    if False:
        for i in range(10):
            print('nop')
    'Extracts html elements from the output of an executed notebook node and\n  writes them into a file sink.'
    if output['output_type'] == 'display_data':
        data = output['data']
        if 'application/javascript' in data:
            sink.write('<script>\n')
            sink.write(data['application/javascript'])
            sink.write('</script>\n')
        if 'text/html' in data:
            parser = IFrameParser()
            parser.feed(data['text/html'])
            if parser.srcdocs:
                sink.write(parser.srcdocs)
            else:
                sink.write(data['text/html'])

class IFrameParser(HTMLParser):
    """A parser to extract iframe content from given HTML."""

    def __init__(self):
        if False:
            print('Hello World!')
        self._srcdocs = []
        super().__init__()

    def handle_starttag(self, tag, attrs):
        if False:
            print('Hello World!')
        if tag == 'iframe':
            for attr in attrs:
                if 'srcdoc' in attr:
                    self._srcdocs.append(html.unescape(attr[1]))

    @property
    def srcdocs(self):
        if False:
            i = 10
            return i + 15
        return '\n'.join(self._srcdocs)