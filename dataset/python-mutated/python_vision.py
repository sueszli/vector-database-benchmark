import ast
import os
import queue
import sys
import threading
import time
import traceback
from jupyter_client import KernelManager
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
os.environ['ANSI_COLORS_DISABLED'] = '1'

class PythonVision:
    file_extension = 'py'
    proper_name = 'Python'

    def __init__(self, config):
        if False:
            while True:
                i = 10
        self.config = config
        self.km = KernelManager(kernel_name='python3')
        self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()
        while not self.kc.is_alive():
            time.sleep(0.1)
        time.sleep(0.5)

    def terminate(self):
        if False:
            while True:
                i = 10
        self.kc.stop_channels()
        self.km.shutdown_kernel()

    def run(self, code):
        if False:
            for i in range(10):
                print('nop')
        preprocessed_code = self.preprocess_code(code)
        message_queue = queue.Queue()
        self._execute_code(preprocessed_code, message_queue)
        return self._capture_output(message_queue)

    def _execute_code(self, code, message_queue):
        if False:
            print('Hello World!')

        def iopub_message_listener():
            if False:
                i = 10
                return i + 15
            while True:
                try:
                    msg = self.kc.iopub_channel.get_msg(timeout=0.1)
                    content = msg['content']
                    if msg['msg_type'] == 'stream':

                        def detect_active_line(line):
                            if False:
                                for i in range(10):
                                    print('nop')
                            active_line = None
                            while '##active_line' in line:
                                active_line = int(line.split('##active_line')[1].split('##')[0])
                                line = line.replace('##active_line' + str(active_line) + '##', '')
                            return (line, active_line)
                        (line, active_line) = detect_active_line(content['text'])
                        if active_line:
                            message_queue.put({'active_line': active_line})
                        message_queue.put({'output': line})
                    elif msg['msg_type'] == 'error':
                        message_queue.put({'output': '\n'.join(content['traceback'])})
                    elif msg['msg_type'] in ['display_data', 'execute_result']:
                        data = content['data']
                        if 'image/png' in data:
                            message_queue.put({'image': data['image/png']})
                        elif 'image/jpeg' in data:
                            message_queue.put({'image': data['image/jpeg']})
                        elif 'text/html' in data:
                            message_queue.put({'html': data['text/html']})
                        elif 'text/plain' in data:
                            message_queue.put({'output': data['text/plain']})
                        elif 'application/javascript' in data:
                            message_queue.put({'javascript': data['application/javascript']})
                except queue.Empty:
                    if self.kc.shell_channel.msg_ready():
                        break
        listener_thread = threading.Thread(target=iopub_message_listener)
        listener_thread.start()
        self.kc.execute(code)
        listener_thread.join()

    def _capture_output(self, message_queue):
        if False:
            return 10
        while True:
            if not message_queue.empty():
                yield message_queue.get()
            else:
                time.sleep(0.1)
            try:
                output = message_queue.get(timeout=0.3)
                yield output
            except queue.Empty:
                for _ in range(3):
                    if not message_queue.empty():
                        yield message_queue.get()
                    time.sleep(0.2)
                break

    def _old_capture_output(self, message_queue):
        if False:
            print('Hello World!')
        output = []
        while True:
            try:
                line = message_queue.get_nowait()
                output.append(line)
            except queue.Empty:
                break
        return output

    def preprocess_code(self, code):
        if False:
            i = 10
            return i + 15
        return preprocess_python(code)

def preprocess_python(code):
    if False:
        return 10
    '\n    Add active line markers\n    Wrap in a try except\n    '
    code = add_active_line_prints(code)
    code_lines = code.split('\n')
    code_lines = [c for c in code_lines if c.strip() != '']
    code = '\n'.join(code_lines)
    return code

def add_active_line_prints(code):
    if False:
        while True:
            i = 10
    '\n    Add print statements indicating line numbers to a python string.\n    '
    tree = ast.parse(code)
    transformer = AddLinePrints()
    new_tree = transformer.visit(tree)
    return ast.unparse(new_tree)

class AddLinePrints(ast.NodeTransformer):
    """
    Transformer to insert print statements indicating the line number
    before every executable line in the AST.
    """

    def insert_print_statement(self, line_number):
        if False:
            print('Hello World!')
        'Inserts a print statement for a given line number.'
        return ast.Expr(value=ast.Call(func=ast.Name(id='print', ctx=ast.Load()), args=[ast.Constant(value=f'##active_line{line_number}##')], keywords=[]))

    def process_body(self, body):
        if False:
            for i in range(10):
                print('nop')
        'Processes a block of statements, adding print calls.'
        new_body = []
        if not isinstance(body, list):
            body = [body]
        for sub_node in body:
            if hasattr(sub_node, 'lineno'):
                new_body.append(self.insert_print_statement(sub_node.lineno))
            new_body.append(sub_node)
        return new_body

    def visit(self, node):
        if False:
            i = 10
            return i + 15
        'Overridden visit to transform nodes.'
        new_node = super().visit(node)
        if hasattr(new_node, 'body'):
            new_node.body = self.process_body(new_node.body)
        if hasattr(new_node, 'orelse') and new_node.orelse:
            new_node.orelse = self.process_body(new_node.orelse)
        if isinstance(new_node, ast.Try):
            for handler in new_node.handlers:
                handler.body = self.process_body(handler.body)
            if new_node.finalbody:
                new_node.finalbody = self.process_body(new_node.finalbody)
        return new_node

def wrap_in_try_except(code):
    if False:
        for i in range(10):
            print('nop')
    code = 'import traceback\n' + code
    parsed_code = ast.parse(code)
    try_except = ast.Try(body=parsed_code.body, handlers=[ast.ExceptHandler(type=ast.Name(id='Exception', ctx=ast.Load()), name=None, body=[ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id='traceback', ctx=ast.Load()), attr='print_exc', ctx=ast.Load()), args=[], keywords=[]))])], orelse=[], finalbody=[])
    parsed_code.body = [try_except]
    return ast.unparse(parsed_code)
'\nconfig = {}  # Your configuration here\npython_kernel = Python(config)\n\ncode = """\nimport pandas as pd\nimport numpy as np\ndf = pd.DataFrame(np.random.rand(10, 5))\n# For HTML output\ndisplay(df)\n# For image output using matplotlib\nimport matplotlib.pyplot as plt\nplt.figure()\nplt.plot(df)\nplt.savefig(\'plot.png\')  # Save the plot as a .png file\nplt.show()\n"""\noutput = python_kernel.run(code)\nfor line in output:\n    display_output(line)\n'