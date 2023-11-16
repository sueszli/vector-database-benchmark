import asyncio
import io
import inspect
import logging
import os
import queue
import uuid
import re
import sys
import threading
import time
import traceback
from typing_extensions import Literal
from werkzeug.serving import make_server
try:
    from IPython import get_ipython
    from IPython.display import IFrame, display, Javascript
    from IPython.core.display import HTML
    from IPython.core.ultratb import FormattedTB
    from retrying import retry
    from ansi2html import Ansi2HTMLConverter
    from ipykernel.comm import Comm
    import nest_asyncio
    import requests
    _dash_comm = Comm(target_name='dash')
    _dep_installed = True
except ImportError:
    _dep_installed = False
    _dash_comm = None
    get_ipython = lambda : None
JupyterDisplayMode = Literal['inline', 'external', 'jupyterlab', 'tab', '_none']

def _get_skip(error: Exception):
    if False:
        for i in range(10):
            print('nop')
    tb = traceback.format_exception(type(error), error, error.__traceback__)
    skip = 0
    for (i, line) in enumerate(tb):
        if '%% callback invoked %%' in line:
            skip = i + 1
            break
    return skip

def _custom_formatargvalues(args, varargs, varkw, locals, formatarg=str, formatvarargs=lambda name: '*' + name, formatvarkw=lambda name: '**' + name, formatvalue=lambda value: '=' + repr(value)):
    if False:
        for i in range(10):
            print('nop')
    'Copied from inspect.formatargvalues, modified to place function\n    arguments on separate lines'

    def convert(name, locals=locals, formatarg=formatarg, formatvalue=formatvalue):
        if False:
            print('Hello World!')
        return formatarg(name) + formatvalue(locals[name])
    specs = []
    for i in range(len(args)):
        specs.append(convert(args[i]))
    if varargs:
        specs.append(formatvarargs(varargs) + formatvalue(locals[varargs]))
    if varkw:
        specs.append(formatvarkw(varkw) + formatvalue(locals[varkw]))
    result = '(' + ', '.join(specs) + ')'
    if len(result) < 40:
        return result
    return '(\n    ' + ',\n    '.join(specs) + '\n)'
_jupyter_config = {}
_caller = {}

def _send_jupyter_config_comm_request():
    if False:
        i = 10
        return i + 15
    if get_ipython() is not None:
        if _dash_comm.kernel is not None:
            _caller['parent'] = _dash_comm.kernel.get_parent()
            _dash_comm.send({'type': 'base_url_request'})

def _jupyter_comm_response_received():
    if False:
        print('Hello World!')
    return bool(_jupyter_config)

def _request_jupyter_config(timeout=2):
    if False:
        i = 10
        return i + 15
    if _dash_comm.kernel is None:
        return
    _send_jupyter_config_comm_request()
    shell = get_ipython()
    kernel = shell.kernel
    captured_events = []

    def capture_event(stream, ident, parent):
        if False:
            i = 10
            return i + 15
        captured_events.append((stream, ident, parent))
    kernel.shell_handlers['execute_request'] = capture_event
    shell.execution_count += 1
    t0 = time.time()
    while True:
        if time.time() - t0 > timeout:
            raise EnvironmentError('Unable to communicate with the jupyter_dash notebook or JupyterLab \nextension required to infer Jupyter configuration.')
        if _jupyter_comm_response_received():
            break
        if asyncio.iscoroutinefunction(kernel.do_one_iteration):
            loop = asyncio.get_event_loop()
            nest_asyncio.apply(loop)
            loop.run_until_complete(kernel.do_one_iteration())
        else:
            kernel.do_one_iteration()
    kernel.shell_handlers['execute_request'] = kernel.execute_request
    sys.stdout.flush()
    sys.stderr.flush()
    for (stream, ident, parent) in captured_events:
        kernel.set_parent(ident, parent)
        kernel.execute_request(stream, ident, parent)

class JupyterDash:
    """
    Interact with dash apps inside jupyter notebooks.
    """
    default_mode: JupyterDisplayMode = 'inline'
    alive_token = str(uuid.uuid4())
    inline_exceptions: bool = True
    _servers = {}

    def infer_jupyter_proxy_config(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Infer the current Jupyter server configuration. This will detect\n        the proper request_pathname_prefix and server_url values to use when\n        displaying Dash apps.Dash requests will be routed through the proxy.\n\n        Requirements:\n\n        In the classic notebook, this method requires the `dash` nbextension\n        which should be installed automatically with the installation of the\n        jupyter-dash Python package. You can see what notebook extensions are installed\n        by running the following command:\n            $ jupyter nbextension list\n\n        In JupyterLab, this method requires the `@plotly/dash-jupyterlab` labextension. This\n        extension should be installed automatically with the installation of the\n        jupyter-dash Python package, but JupyterLab must be allowed to rebuild before\n        the extension is activated (JupyterLab should automatically detect the\n        extension and produce a popup dialog asking for permission to rebuild). You can\n        see what JupyterLab extensions are installed by running the following command:\n            $ jupyter labextension list\n        '
        if not self.in_ipython or self.in_colab:
            return
        _request_jupyter_config()

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.in_ipython = get_ipython() is not None
        self.in_colab = 'google.colab' in sys.modules
        if _dep_installed and self.in_ipython and _dash_comm:

            @_dash_comm.on_msg
            def _receive_message(msg):
                if False:
                    print('Hello World!')
                prev_parent = _caller.get('parent')
                if prev_parent and prev_parent != _dash_comm.kernel.get_parent():
                    _dash_comm.kernel.set_parent([prev_parent['header']['session']], prev_parent)
                    del _caller['parent']
                msg_data = msg.get('content').get('data')
                msg_type = msg_data.get('type', None)
                if msg_type == 'base_url_response':
                    _jupyter_config.update(msg_data)

    def run_app(self, app, mode: JupyterDisplayMode=None, width='100%', height=650, host='127.0.0.1', port=8050, server_url=None):
        if False:
            print('Hello World!')
        '\n        :type app: dash.Dash\n        :param mode: How to display the app on the notebook. One Of:\n            ``"external"``: The URL of the app will be displayed in the notebook\n                output cell. Clicking this URL will open the app in the default\n                web browser.\n            ``"inline"``: The app will be displayed inline in the notebook output cell\n                in an iframe.\n            ``"jupyterlab"``: The app will be displayed in a dedicate tab in the\n                JupyterLab interface. Requires JupyterLab and the `jupyterlab-dash`\n                extension.\n        :param width: Width of app when displayed using mode="inline"\n        :param height: Height of app when displayed using mode="inline"\n        :param host: Host of the server\n        :param port: Port used by the server\n        :param server_url: Use if a custom url is required to display the app.\n        '
        if self.in_colab:
            valid_display_values = ['inline', 'external']
        else:
            valid_display_values = ['jupyterlab', 'inline', 'external', 'tab', '_none']
        if mode is None:
            mode = self.default_mode
        elif not isinstance(mode, str):
            raise ValueError(f'The mode argument must be a string\n    Received value of type {type(mode)}: {repr(mode)}')
        else:
            mode = mode.lower()
            if mode not in valid_display_values:
                raise ValueError(f'Invalid display argument {mode}\n    Valid arguments: {valid_display_values}')
        old_server = self._servers.get((host, port))
        if old_server:
            old_server.shutdown()
            del self._servers[host, port]
        if 'base_subpath' in _jupyter_config:
            requests_pathname_prefix = _jupyter_config['base_subpath'].rstrip('/') + '/proxy/{port}/'
        else:
            requests_pathname_prefix = app.config.get('requests_pathname_prefix', None)
        if requests_pathname_prefix is not None:
            requests_pathname_prefix = requests_pathname_prefix.format(port=port)
        else:
            requests_pathname_prefix = '/'
        dict.__setitem__(app.config, 'requests_pathname_prefix', requests_pathname_prefix)
        if server_url is None:
            if 'server_url' in _jupyter_config:
                server_url = _jupyter_config['server_url'].rstrip('/')
            else:
                domain_base = os.environ.get('DASH_DOMAIN_BASE', None)
                if domain_base:
                    server_url = 'https://' + domain_base
                else:
                    server_url = f'http://{host}:{port}'
        else:
            server_url = server_url.rstrip('/')
        dashboard_url = f'{server_url}{requests_pathname_prefix}'
        try:
            import orjson
        except ImportError:
            pass
        err_q = queue.Queue()
        server = make_server(host, port, app.server, threaded=True, processes=0)
        logging.getLogger('werkzeug').setLevel(logging.ERROR)

        @retry(stop_max_attempt_number=15, wait_exponential_multiplier=100, wait_exponential_max=1000)
        def run():
            if False:
                while True:
                    i = 10
            try:
                server.serve_forever()
            except SystemExit:
                pass
            except Exception as error:
                err_q.put(error)
                raise error
        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()
        self._servers[host, port] = server
        alive_url = f'http://{host}:{port}/_alive_{JupyterDash.alive_token}'

        def _get_error():
            if False:
                print('Hello World!')
            try:
                err = err_q.get_nowait()
                if err:
                    raise err
            except queue.Empty:
                pass

        @retry(stop_max_attempt_number=15, wait_exponential_multiplier=10, wait_exponential_max=1000)
        def wait_for_app():
            if False:
                return 10
            _get_error()
            try:
                req = requests.get(alive_url)
                res = req.content.decode()
                if req.status_code != 200:
                    raise Exception(res)
                if res != 'Alive':
                    url = f'http://{host}:{port}'
                    raise OSError(f"Address '{url}' already in use.\n    Try passing a different port to run_server.")
            except requests.ConnectionError as err:
                _get_error()
                raise err
        try:
            wait_for_app()
            if self.in_colab:
                JupyterDash._display_in_colab(dashboard_url, port, mode, width, height)
            else:
                JupyterDash._display_in_jupyter(dashboard_url, port, mode, width, height)
        except Exception as final_error:
            msg = str(final_error)
            if msg.startswith('<!'):
                display(HTML(msg))
            else:
                raise final_error

    @staticmethod
    def _display_in_colab(dashboard_url, port, mode, width, height):
        if False:
            for i in range(10):
                print('nop')
        from google.colab import output
        if mode == 'inline':
            output.serve_kernel_port_as_iframe(port, width=width, height=height)
        elif mode == 'external':
            print('Dash app running on:')
            output.serve_kernel_port_as_window(port, anchor_text=dashboard_url)

    @staticmethod
    def _display_in_jupyter(dashboard_url, port, mode, width, height):
        if False:
            return 10
        if mode == 'inline':
            display(IFrame(dashboard_url, width, height))
        elif mode in ('external', 'tab'):
            print(f'Dash app running on {dashboard_url}')
            if mode == 'tab':
                display(Javascript(f"window.open('{dashboard_url}')"))
        elif mode == 'jupyterlab':
            _dash_comm.send({'type': 'show', 'port': port, 'url': dashboard_url})

    @staticmethod
    def serve_alive():
        if False:
            for i in range(10):
                print('nop')
        return 'Alive'

    def configure_callback_exception_handling(self, app, dev_tools_prune_errors):
        if False:
            print('Hello World!')
        'Install traceback handling for callbacks'

        @app.server.errorhandler(Exception)
        def _wrap_errors(error):
            if False:
                i = 10
                return i + 15
            skip = _get_skip(error) if dev_tools_prune_errors else 0
            original_formatargvalues = inspect.formatargvalues
            inspect.formatargvalues = _custom_formatargvalues
            try:
                ostream = io.StringIO()
                ipytb = FormattedTB(tb_offset=skip, mode='Verbose', color_scheme='Linux', include_vars=True, ostream=ostream)
                ipytb()
            finally:
                inspect.formatargvalues = original_formatargvalues
            ansi_stacktrace = ostream.getvalue()
            if self.inline_exceptions:
                print(ansi_stacktrace)
            conv = Ansi2HTMLConverter(scheme='ansi2html', dark_bg=False)
            html_str = conv.convert(ansi_stacktrace)
            html_str = html_str.replace('<html>', '<html style="width: 75ch; font-size: 0.86em">')
            html_str = re.sub('background-color:[^;]+;', '', html_str)
            return (html_str, 500)

    @property
    def active(self):
        if False:
            i = 10
            return i + 15
        _inside_dbx = 'DATABRICKS_RUNTIME_VERSION' in os.environ
        return _dep_installed and (not _inside_dbx) and (self.in_ipython or self.in_colab)
jupyter_dash = JupyterDash()