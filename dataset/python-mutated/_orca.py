import atexit
import json
import os
import socket
import subprocess
import sys
import threading
import warnings
from copy import copy
from contextlib import contextmanager
from pathlib import Path
from shutil import which
import tenacity
import plotly
from plotly.files import PLOTLY_DIR, ensure_writable_plotly_dir
from plotly.io._utils import validate_coerce_fig_to_dict
from plotly.optional_imports import get_module
psutil = get_module('psutil')
valid_formats = ('png', 'jpeg', 'webp', 'svg', 'pdf', 'eps')
format_conversions = {fmt: fmt for fmt in valid_formats}
format_conversions.update({'jpg': 'jpeg'})

def raise_format_value_error(val):
    if False:
        print('Hello World!')
    raise ValueError('\nInvalid value of type {typ} receive as an image format specification.\n    Received value: {v}\n\nAn image format must be specified as one of the following string values:\n    {valid_formats}'.format(typ=type(val), v=val, valid_formats=sorted(format_conversions.keys())))

def validate_coerce_format(fmt):
    if False:
        i = 10
        return i + 15
    "\n    Validate / coerce a user specified image format, and raise an informative\n    exception if format is invalid.\n\n    Parameters\n    ----------\n    fmt\n        A value that may or may not be a valid image format string.\n\n    Returns\n    -------\n    str or None\n        A valid image format string as supported by orca. This may not\n        be identical to the input image designation. For example,\n        the resulting string will always be lower case and  'jpg' is\n        converted to 'jpeg'.\n\n        If the input format value is None, then no exception is raised and\n        None is returned.\n\n    Raises\n    ------\n    ValueError\n        if the input `fmt` cannot be interpreted as a valid image format.\n    "
    if fmt is None:
        return None
    if not isinstance(fmt, str) or not fmt:
        raise_format_value_error(fmt)
    fmt = fmt.lower()
    if fmt[0] == '.':
        fmt = fmt[1:]
    if fmt not in format_conversions:
        raise_format_value_error(fmt)
    return format_conversions[fmt]

def find_open_port():
    if False:
        for i in range(10):
            print('nop')
    '\n    Use the socket module to find an open port.\n\n    Returns\n    -------\n    int\n        An open port\n    '
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('', 0))
    (_, port) = s.getsockname()
    s.close()
    return port

class OrcaConfig(object):
    """
    Singleton object containing the current user defined configuration
    properties for orca.

    These parameters may optionally be saved to the user's ~/.plotly
    directory using the `save` method, in which case they are automatically
    restored in future sessions.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._props = {}
        root_dir = os.path.dirname(os.path.abspath(plotly.__file__))
        self.package_dir = os.path.join(root_dir, 'package_data')
        self.reload(warn=False)
        plotlyjs = os.path.join(self.package_dir, 'plotly.min.js')
        self._constants = {'plotlyjs': plotlyjs, 'config_file': os.path.join(PLOTLY_DIR, '.orca')}

    def restore_defaults(self, reset_server=True):
        if False:
            return 10
        '\n        Reset all orca configuration properties to their default values\n        '
        self._props = {}
        if reset_server:
            reset_status()

    def update(self, d={}, **kwargs):
        if False:
            print('Hello World!')
        "\n        Update one or more properties from a dict or from input keyword\n        arguments.\n\n        Parameters\n        ----------\n        d: dict\n            Dictionary from property names to new property values.\n\n        kwargs\n            Named argument value pairs where the name is a configuration\n            property name and the value is the new property value.\n\n        Returns\n        -------\n        None\n\n        Examples\n        --------\n        Update configuration properties using a dictionary\n\n        >>> import plotly.io as pio\n        >>> pio.orca.config.update({'timeout': 30, 'default_format': 'svg'})\n\n        Update configuration properties using keyword arguments\n\n        >>> pio.orca.config.update(timeout=30, default_format='svg'})\n        "
        if not isinstance(d, dict):
            raise ValueError('\nThe first argument to update must be a dict, but received value of type {typ}l\n    Received value: {val}'.format(typ=type(d), val=d))
        updates = copy(d)
        updates.update(kwargs)
        for k in updates:
            if k not in self._props:
                raise ValueError('Invalid property name: {k}'.format(k=k))
        for (k, v) in updates.items():
            setattr(self, k, v)

    def reload(self, warn=True):
        if False:
            i = 10
            return i + 15
        '\n        Reload orca settings from ~/.plotly/.orca, if any.\n\n        Note: Settings are loaded automatically when plotly is imported.\n        This method is only needed if the setting are changed by some outside\n        process (e.g. a text editor) during an interactive session.\n\n        Parameters\n        ----------\n        warn: bool\n            If True, raise informative warnings if settings cannot be restored.\n            If False, do not raise warnings if setting cannot be restored.\n\n        Returns\n        -------\n        None\n        '
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    orca_str = f.read()
            except:
                if warn:
                    warnings.warn('Unable to read orca configuration file at {path}'.format(path=self.config_file))
                return
            try:
                orca_props = json.loads(orca_str)
            except ValueError:
                if warn:
                    warnings.warn('Orca configuration file at {path} is not valid JSON'.format(path=self.config_file))
                return
            for (k, v) in orca_props.items():
                self._props[k] = v
        elif warn:
            warnings.warn('Orca configuration file at {path} not found'.format(path=self.config_file))

    def save(self):
        if False:
            print('Hello World!')
        '\n        Attempt to save current settings to disk, so that they are\n        automatically restored for future sessions.\n\n        This operation requires write access to the path returned by\n        in the `config_file` property.\n\n        Returns\n        -------\n        None\n        '
        if ensure_writable_plotly_dir():
            with open(self.config_file, 'w') as f:
                json.dump(self._props, f, indent=4)
        else:
            warnings.warn("Failed to write orca configuration file at '{path}'".format(path=self.config_file))

    @property
    def server_url(self):
        if False:
            return 10
        '\n        The server URL to use for an external orca server, or None if orca\n        should be managed locally\n\n        Overrides executable, port, timeout, mathjax, topojson,\n        and mapbox_access_token\n\n        Returns\n        -------\n        str or None\n        '
        return self._props.get('server_url', None)

    @server_url.setter
    def server_url(self, val):
        if False:
            return 10
        if val is None:
            self._props.pop('server_url', None)
            return
        if not isinstance(val, str):
            raise ValueError('\nThe server_url property must be a string, but received value of type {typ}.\n    Received value: {val}'.format(typ=type(val), val=val))
        if not val.startswith('http://') and (not val.startswith('https://')):
            val = 'http://' + val
        shutdown_server()
        self.executable = None
        self.port = None
        self.timeout = None
        self.mathjax = None
        self.topojson = None
        self.mapbox_access_token = None
        self._props['server_url'] = val

    @property
    def port(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The specific port to use to communicate with the orca server, or\n        None if the port is to be chosen automatically.\n\n        If an orca server is active, the port in use is stored in the\n        plotly.io.orca.status.port property.\n\n        Returns\n        -------\n        int or None\n        '
        return self._props.get('port', None)

    @port.setter
    def port(self, val):
        if False:
            for i in range(10):
                print('nop')
        if val is None:
            self._props.pop('port', None)
            return
        if not isinstance(val, int):
            raise ValueError('\nThe port property must be an integer, but received value of type {typ}.\n    Received value: {val}'.format(typ=type(val), val=val))
        self._props['port'] = val

    @property
    def executable(self):
        if False:
            while True:
                i = 10
        "\n        The name or full path of the orca executable.\n\n         - If a name (e.g. 'orca'), then it should be the name of an orca\n           executable on the PATH. The directories on the PATH can be\n           displayed by running the following command:\n\n           >>> import os\n           >>> print(os.environ.get('PATH').replace(os.pathsep, os.linesep))\n\n         - If a full path (e.g. '/path/to/orca'), then\n           it should be the full path to an orca executable. In this case\n           the executable does not need to reside on the PATH.\n\n        If an orca server has been validated, then the full path to the\n        validated orca executable is stored in the\n        plotly.io.orca.status.executable property.\n\n        Returns\n        -------\n        str\n        "
        executable_list = self._props.get('executable_list', ['orca'])
        if executable_list is None:
            return None
        else:
            return ' '.join(executable_list)

    @executable.setter
    def executable(self, val):
        if False:
            return 10
        if val is None:
            self._props.pop('executable', None)
        else:
            if not isinstance(val, str):
                raise ValueError('\nThe executable property must be a string, but received value of type {typ}.\n    Received value: {val}'.format(typ=type(val), val=val))
            if isinstance(val, str):
                val = [val]
            self._props['executable_list'] = val
        reset_status()

    @property
    def timeout(self):
        if False:
            print('Hello World!')
        '\n        The number of seconds of inactivity required before the orca server\n        is shut down.\n\n        For example, if timeout is set to 20, then the orca\n        server will shutdown once is has not been used for at least\n        20 seconds. If timeout is set to None, then the server will not be\n        automatically shut down due to inactivity.\n\n        Regardless of the value of timeout, a running orca server may be\n        manually shut down like this:\n\n        >>> import plotly.io as pio\n        >>> pio.orca.shutdown_server()\n\n        Returns\n        -------\n        int or float or None\n        '
        return self._props.get('timeout', None)

    @timeout.setter
    def timeout(self, val):
        if False:
            while True:
                i = 10
        if val is None:
            self._props.pop('timeout', None)
        else:
            if not isinstance(val, (int, float)):
                raise ValueError('\nThe timeout property must be a number, but received value of type {typ}.\n    Received value: {val}'.format(typ=type(val), val=val))
            self._props['timeout'] = val
        shutdown_server()

    @property
    def default_width(self):
        if False:
            while True:
                i = 10
        '\n        The default width to use on image export. This value is only\n        applied if no width value is supplied to the plotly.io\n        to_image or write_image functions.\n\n        Returns\n        -------\n        int or None\n        '
        return self._props.get('default_width', None)

    @default_width.setter
    def default_width(self, val):
        if False:
            for i in range(10):
                print('nop')
        if val is None:
            self._props.pop('default_width', None)
            return
        if not isinstance(val, int):
            raise ValueError('\nThe default_width property must be an int, but received value of type {typ}.\n    Received value: {val}'.format(typ=type(val), val=val))
        self._props['default_width'] = val

    @property
    def default_height(self):
        if False:
            i = 10
            return i + 15
        '\n        The default height to use on image export. This value is only\n        applied if no height value is supplied to the plotly.io\n        to_image or write_image functions.\n\n        Returns\n        -------\n        int or None\n        '
        return self._props.get('default_height', None)

    @default_height.setter
    def default_height(self, val):
        if False:
            return 10
        if val is None:
            self._props.pop('default_height', None)
            return
        if not isinstance(val, int):
            raise ValueError('\nThe default_height property must be an int, but received value of type {typ}.\n    Received value: {val}'.format(typ=type(val), val=val))
        self._props['default_height'] = val

    @property
    def default_format(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The default image format to use on image export.\n\n        Valid image formats strings are:\n          - 'png'\n          - 'jpg' or 'jpeg'\n          - 'webp'\n          - 'svg'\n          - 'pdf'\n          - 'eps' (Requires the poppler library to be installed)\n\n        This value is only applied if no format value is supplied to the\n        plotly.io to_image or write_image functions.\n\n        Returns\n        -------\n        str or None\n        "
        return self._props.get('default_format', 'png')

    @default_format.setter
    def default_format(self, val):
        if False:
            while True:
                i = 10
        if val is None:
            self._props.pop('default_format', None)
            return
        val = validate_coerce_format(val)
        self._props['default_format'] = val

    @property
    def default_scale(self):
        if False:
            while True:
                i = 10
        '\n        The default image scaling factor to use on image export.\n        This value is only applied if no scale value is supplied to the\n        plotly.io to_image or write_image functions.\n\n        Returns\n        -------\n        int or None\n        '
        return self._props.get('default_scale', 1)

    @default_scale.setter
    def default_scale(self, val):
        if False:
            i = 10
            return i + 15
        if val is None:
            self._props.pop('default_scale', None)
            return
        if not isinstance(val, (int, float)):
            raise ValueError('\nThe default_scale property must be a number, but received value of type {typ}.\n    Received value: {val}'.format(typ=type(val), val=val))
        self._props['default_scale'] = val

    @property
    def topojson(self):
        if False:
            while True:
                i = 10
        '\n        Path to the topojson files needed to render choropleth traces.\n\n        If None, topojson files from the plot.ly CDN are used.\n\n        Returns\n        -------\n        str\n        '
        return self._props.get('topojson', None)

    @topojson.setter
    def topojson(self, val):
        if False:
            print('Hello World!')
        if val is None:
            self._props.pop('topojson', None)
        else:
            if not isinstance(val, str):
                raise ValueError('\nThe topojson property must be a string, but received value of type {typ}.\n    Received value: {val}'.format(typ=type(val), val=val))
            self._props['topojson'] = val
        shutdown_server()

    @property
    def mathjax(self):
        if False:
            print('Hello World!')
        '\n        Path to the MathJax bundle needed to render LaTeX characters\n\n        Returns\n        -------\n        str\n        '
        return self._props.get('mathjax', 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js')

    @mathjax.setter
    def mathjax(self, val):
        if False:
            while True:
                i = 10
        if val is None:
            self._props.pop('mathjax', None)
        else:
            if not isinstance(val, str):
                raise ValueError('\nThe mathjax property must be a string, but received value of type {typ}.\n    Received value: {val}'.format(typ=type(val), val=val))
            self._props['mathjax'] = val
        shutdown_server()

    @property
    def mapbox_access_token(self):
        if False:
            while True:
                i = 10
        '\n        Mapbox access token required to render mapbox traces.\n\n        Returns\n        -------\n        str\n        '
        return self._props.get('mapbox_access_token', None)

    @mapbox_access_token.setter
    def mapbox_access_token(self, val):
        if False:
            for i in range(10):
                print('nop')
        if val is None:
            self._props.pop('mapbox_access_token', None)
        else:
            if not isinstance(val, str):
                raise ValueError('\nThe mapbox_access_token property must be a string, but received value of type {typ}.\n    Received value: {val}'.format(typ=type(val), val=val))
            self._props['mapbox_access_token'] = val
        shutdown_server()

    @property
    def use_xvfb(self):
        if False:
            print('Hello World!')
        dflt = 'auto'
        return self._props.get('use_xvfb', dflt)

    @use_xvfb.setter
    def use_xvfb(self, val):
        if False:
            while True:
                i = 10
        valid_vals = [True, False, 'auto']
        if val is None:
            self._props.pop('use_xvfb', None)
        else:
            if val not in valid_vals:
                raise ValueError('\nThe use_xvfb property must be one of {valid_vals}\n    Received value of type {typ}: {val}'.format(valid_vals=valid_vals, typ=type(val), val=repr(val)))
            self._props['use_xvfb'] = val
        reset_status()

    @property
    def plotlyjs(self):
        if False:
            i = 10
            return i + 15
        '\n        The plotly.js bundle being used for image rendering.\n\n        Returns\n        -------\n        str\n        '
        return self._constants.get('plotlyjs', None)

    @property
    def config_file(self):
        if False:
            return 10
        '\n        Path to orca configuration file\n\n        Using the `plotly.io.config.save()` method will save the current\n        configuration settings to this file. Settings in this file are\n        restored at the beginning of each sessions.\n\n        Returns\n        -------\n        str\n        '
        return os.path.join(PLOTLY_DIR, '.orca')

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        '\n        Display a nice representation of the current orca configuration.\n        '
        return 'orca configuration\n------------------\n    server_url: {server_url}\n    executable: {executable}\n    port: {port}\n    timeout: {timeout}\n    default_width: {default_width}\n    default_height: {default_height}\n    default_scale: {default_scale}\n    default_format: {default_format}\n    mathjax: {mathjax}\n    topojson: {topojson}\n    mapbox_access_token: {mapbox_access_token}\n    use_xvfb: {use_xvfb}\n\nconstants\n---------\n    plotlyjs: {plotlyjs}\n    config_file: {config_file}\n\n'.format(server_url=self.server_url, port=self.port, executable=self.executable, timeout=self.timeout, default_width=self.default_width, default_height=self.default_height, default_scale=self.default_scale, default_format=self.default_format, mathjax=self.mathjax, topojson=self.topojson, mapbox_access_token=self.mapbox_access_token, plotlyjs=self.plotlyjs, config_file=self.config_file, use_xvfb=self.use_xvfb)
config = OrcaConfig()
del OrcaConfig

class OrcaStatus(object):
    """
    Class to store information about the current status of the orca server.
    """
    _props = {'state': 'unvalidated', 'executable_list': None, 'version': None, 'pid': None, 'port': None, 'command': None}

    @property
    def state(self):
        if False:
            while True:
                i = 10
        '\n        A string representing the state of the orca server process\n\n        One of:\n          - unvalidated: The orca executable has not yet been searched for or\n            tested to make sure its valid.\n          - validated: The orca executable has been located and tested for\n            validity, but it is not running.\n          - running: The orca server process is currently running.\n        '
        return self._props['state']

    @property
    def executable(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        If the `state` property is 'validated' or 'running', this property\n        contains the full path to the orca executable.\n\n        This path can be specified explicitly by setting the `executable`\n        property of the `plotly.io.orca.config` object.\n\n        This property will be None if the `state` is 'unvalidated'.\n        "
        executable_list = self._props['executable_list']
        if executable_list is None:
            return None
        else:
            return ' '.join(executable_list)

    @property
    def version(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        If the `state` property is 'validated' or 'running', this property\n        contains the version of the validated orca executable.\n\n        This property will be None if the `state` is 'unvalidated'.\n        "
        return self._props['version']

    @property
    def pid(self):
        if False:
            return 10
        "\n        The process id of the orca server process, if any. This property\n        will be None if the `state` is not 'running'.\n        "
        return self._props['pid']

    @property
    def port(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The port number that the orca server process is listening to, if any.\n        This property will be None if the `state` is not 'running'.\n\n        This port can be specified explicitly by setting the `port`\n        property of the `plotly.io.orca.config` object.\n        "
        return self._props['port']

    @property
    def command(self):
        if False:
            while True:
                i = 10
        "\n        The command arguments used to launch the running orca server, if any.\n        This property will be None if the `state` is not 'running'.\n        "
        return self._props['command']

    def __repr__(self):
        if False:
            while True:
                i = 10
        '\n        Display a nice representation of the current orca server status.\n        '
        return 'orca status\n-----------\n    state: {state}\n    executable: {executable}\n    version: {version}\n    port: {port}\n    pid: {pid}\n    command: {command}\n\n'.format(executable=self.executable, version=self.version, port=self.port, pid=self.pid, state=self.state, command=self.command)
status = OrcaStatus()
del OrcaStatus

@contextmanager
def orca_env():
    if False:
        return 10
    '\n    Context manager to clear and restore environment variables that are\n    problematic for orca to function properly\n\n    NODE_OPTIONS: When this variable is set, orca <v1.2 will have a\n    segmentation fault due to an electron bug.\n    See: https://github.com/electron/electron/issues/12695\n\n    ELECTRON_RUN_AS_NODE: When this environment variable is set the call\n    to orca is transformed into a call to nodejs.\n    See https://github.com/plotly/orca/issues/149#issuecomment-443506732\n    '
    clear_env_vars = ['NODE_OPTIONS', 'ELECTRON_RUN_AS_NODE', 'LD_PRELOAD']
    orig_env_vars = {}
    try:
        orig_env_vars.update({var: os.environ.pop(var) for var in clear_env_vars if var in os.environ})
        yield
    finally:
        for (var, val) in orig_env_vars.items():
            os.environ[var] = val

def validate_executable():
    if False:
        i = 10
        return i + 15
    "\n    Attempt to find and validate the orca executable specified by the\n    `plotly.io.orca.config.executable` property.\n\n    If the `plotly.io.orca.status.state` property is 'validated' or 'running'\n    then this function does nothing.\n\n    How it works:\n      - First, it searches the system PATH for an executable that matches the\n      name or path specified in the `plotly.io.orca.config.executable`\n      property.\n      - Then it runs the executable with the `--help` flag to make sure\n      it's the plotly orca executable\n      - Then it runs the executable with the `--version` flag to check the\n      orca version.\n\n    If all of these steps are successful then the `status.state` property\n    is set to 'validated' and the `status.executable` and `status.version`\n    properties are populated\n\n    Returns\n    -------\n    None\n    "
    if status.state != 'unvalidated':
        return
    install_location_instructions = "If you haven't installed orca yet, you can do so using conda as follows:\n\n    $ conda install -c plotly plotly-orca\n\nAlternatively, see other installation methods in the orca project README at\nhttps://github.com/plotly/orca\n\nAfter installation is complete, no further configuration should be needed.\n\nIf you have installed orca, then for some reason plotly.py was unable to\nlocate it. In this case, set the `plotly.io.orca.config.executable`\nproperty to the full path of your orca executable. For example:\n\n    >>> plotly.io.orca.config.executable = '/path/to/orca'\n\nAfter updating this executable property, try the export operation again.\nIf it is successful then you may want to save this configuration so that it\nwill be applied automatically in future sessions. You can do this as follows:\n\n    >>> plotly.io.orca.config.save()\n\nIf you're still having trouble, feel free to ask for help on the forums at\nhttps://community.plot.ly/c/api/python\n"
    executable = which(config.executable)
    path = os.environ.get('PATH', os.defpath)
    formatted_path = path.replace(os.pathsep, '\n    ')
    if executable is None:
        raise ValueError("\nThe orca executable is required to export figures as static images,\nbut it could not be found on the system path.\n\nSearched for executable '{executable}' on the following path:\n    {formatted_path}\n\n{instructions}".format(executable=config.executable, formatted_path=formatted_path, instructions=install_location_instructions))
    xvfb_args = ['--auto-servernum', '--server-args', '-screen 0 640x480x24 +extension RANDR +extension GLX', executable]
    if config.use_xvfb == True:
        xvfb_run_executable = which('xvfb-run')
        if not xvfb_run_executable:
            raise ValueError("\nThe plotly.io.orca.config.use_xvfb property is set to True, but the\nxvfb-run executable could not be found on the system path.\n\nSearched for the executable 'xvfb-run' on the following path:\n    {formatted_path}".format(formatted_path=formatted_path))
        executable_list = [xvfb_run_executable] + xvfb_args
    elif config.use_xvfb == 'auto' and sys.platform.startswith('linux') and (not os.environ.get('DISPLAY')) and which('xvfb-run'):
        xvfb_run_executable = which('xvfb-run')
        executable_list = [xvfb_run_executable] + xvfb_args
    else:
        executable_list = [executable]
    invalid_executable_msg = "\nThe orca executable is required in order to export figures as static images,\nbut the executable that was found at '{executable}'\ndoes not seem to be a valid plotly orca executable. Please refer to the end of\nthis message for details on what went wrong.\n\n{instructions}".format(executable=executable, instructions=install_location_instructions)
    with orca_env():
        p = subprocess.Popen(executable_list + ['--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (help_result, help_error) = p.communicate()
    if p.returncode != 0:
        err_msg = invalid_executable_msg + '\nHere is the error that was returned by the command\n    $ {executable} --help\n\n[Return code: {returncode}]\n{err_msg}\n'.format(executable=' '.join(executable_list), err_msg=help_error.decode('utf-8'), returncode=p.returncode)
        if sys.platform.startswith('linux') and (not os.environ.get('DISPLAY')):
            err_msg += 'Note: When used on Linux, orca requires an X11 display server, but none was\ndetected. Please install Xvfb and configure plotly.py to run orca using Xvfb\nas follows:\n\n    >>> import plotly.io as pio\n    >>> pio.orca.config.use_xvfb = True\n\nYou can save this configuration for use in future sessions as follows:\n\n    >>> pio.orca.config.save()\n\nSee https://www.x.org/releases/X11R7.6/doc/man/man1/Xvfb.1.xhtml\nfor more info on Xvfb\n'
        raise ValueError(err_msg)
    if not help_result:
        raise ValueError(invalid_executable_msg + '\nThe error encountered is that no output was returned by the command\n    $ {executable} --help\n'.format(executable=' '.join(executable_list)))
    if "Plotly's image-exporting utilities" not in help_result.decode('utf-8'):
        raise ValueError(invalid_executable_msg + '\nThe error encountered is that unexpected output was returned by the command\n    $ {executable} --help\n\n{help_result}\n'.format(executable=' '.join(executable_list), help_result=help_result))
    with orca_env():
        p = subprocess.Popen(executable_list + ['--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (version_result, version_error) = p.communicate()
    if p.returncode != 0:
        raise ValueError(invalid_executable_msg + '\nAn error occurred while trying to get the version of the orca executable.\nHere is the command that plotly.py ran to request the version\n    $ {executable} --version\n\nThis command returned the following error:\n\n[Return code: {returncode}]\n{err_msg}\n        '.format(executable=' '.join(executable_list), err_msg=version_error.decode('utf-8'), returncode=p.returncode))
    if not version_result:
        raise ValueError(invalid_executable_msg + '\nThe error encountered is that no version was reported by the orca executable.\nHere is the command that plotly.py ran to request the version:\n\n    $ {executable} --version\n'.format(executable=' '.join(executable_list)))
    else:
        version_result = version_result.decode()
    status._props['executable_list'] = executable_list
    status._props['version'] = version_result.strip()
    status._props['state'] = 'validated'

def reset_status():
    if False:
        while True:
            i = 10
    '\n    Shutdown the running orca server, if any, and reset the orca status\n    to unvalidated.\n\n    This command is only needed if the desired orca executable is changed\n    during an interactive session.\n\n    Returns\n    -------\n    None\n    '
    shutdown_server()
    status._props['executable_list'] = None
    status._props['version'] = None
    status._props['state'] = 'unvalidated'
orca_lock = threading.Lock()
orca_state = {'proc': None, 'shutdown_timer': None}

@atexit.register
def cleanup():
    if False:
        while True:
            i = 10
    shutdown_server()

def shutdown_server():
    if False:
        print('Hello World!')
    '\n    Shutdown the running orca server process, if any\n\n    Returns\n    -------\n    None\n    '
    if orca_state['proc'] is not None:
        with orca_lock:
            if orca_state['proc'] is not None:
                parent = psutil.Process(orca_state['proc'].pid)
                for child in parent.children(recursive=True):
                    try:
                        child.terminate()
                    except:
                        pass
                try:
                    orca_state['proc'].terminate()
                    child_status = orca_state['proc'].wait()
                except:
                    pass
                orca_state['proc'] = None
                if orca_state['shutdown_timer'] is not None:
                    orca_state['shutdown_timer'].cancel()
                    orca_state['shutdown_timer'] = None
                orca_state['port'] = None
                status._props['state'] = 'validated'
                status._props['pid'] = None
                status._props['port'] = None
                status._props['command'] = None

def ensure_server():
    if False:
        return 10
    '\n    Start an orca server if none is running. If a server is already running,\n    then reset the timeout countdown\n\n    Returns\n    -------\n    None\n    '
    if psutil is None:
        raise ValueError('Image generation requires the psutil package.\n\nInstall using pip:\n    $ pip install psutil\n\nInstall using conda:\n    $ conda install psutil\n')
    if not get_module('requests'):
        raise ValueError('Image generation requires the requests package.\n\nInstall using pip:\n    $ pip install requests\n\nInstall using conda:\n    $ conda install requests\n')
    if not config.server_url:
        if status.state == 'unvalidated':
            validate_executable()
        with orca_lock:
            if orca_state['shutdown_timer'] is not None:
                orca_state['shutdown_timer'].cancel()
            if orca_state['proc'] is None:
                if config.port is None:
                    orca_state['port'] = find_open_port()
                else:
                    orca_state['port'] = config.port
                cmd_list = status._props['executable_list'] + ['serve', '-p', str(orca_state['port']), '--plotly', config.plotlyjs, '--graph-only']
                if config.topojson:
                    cmd_list.extend(['--topojson', config.topojson])
                if config.mathjax:
                    cmd_list.extend(['--mathjax', config.mathjax])
                if config.mapbox_access_token:
                    cmd_list.extend(['--mapbox-access-token', config.mapbox_access_token])
                DEVNULL = open(os.devnull, 'wb')
                with orca_env():
                    stderr = DEVNULL if 'CI' in os.environ else None
                    orca_state['proc'] = subprocess.Popen(cmd_list, stdout=DEVNULL, stderr=stderr)
                status._props['state'] = 'running'
                status._props['pid'] = orca_state['proc'].pid
                status._props['port'] = orca_state['port']
                status._props['command'] = cmd_list
            if config.timeout is not None:
                t = threading.Timer(config.timeout, shutdown_server)
                t.daemon = True
                t.start()
                orca_state['shutdown_timer'] = t

@tenacity.retry(wait=tenacity.wait_random(min=5, max=10), stop=tenacity.stop_after_delay(60000))
def request_image_with_retrying(**kwargs):
    if False:
        print('Hello World!')
    '\n    Helper method to perform an image request to a running orca server process\n    with retrying logic.\n    '
    from requests import post
    from plotly.io.json import to_json_plotly
    if config.server_url:
        server_url = config.server_url
    else:
        server_url = 'http://{hostname}:{port}'.format(hostname='localhost', port=orca_state['port'])
    request_params = {k: v for (k, v) in kwargs.items() if v is not None}
    json_str = to_json_plotly(request_params)
    response = post(server_url + '/', data=json_str)
    if response.status_code == 522:
        shutdown_server()
        ensure_server()
        raise OSError('522: client socket timeout')
    return response

def to_image(fig, format=None, width=None, height=None, scale=None, validate=True):
    if False:
        return 10
    "\n    Convert a figure to a static image bytes string\n\n    Parameters\n    ----------\n    fig:\n        Figure object or dict representing a figure\n\n    format: str or None\n        The desired image format. One of\n          - 'png'\n          - 'jpg' or 'jpeg'\n          - 'webp'\n          - 'svg'\n          - 'pdf'\n          - 'eps' (Requires the poppler library to be installed)\n\n        If not specified, will default to `plotly.io.config.default_format`\n\n    width: int or None\n        The width of the exported image in layout pixels. If the `scale`\n        property is 1.0, this will also be the width of the exported image\n        in physical pixels.\n\n        If not specified, will default to `plotly.io.config.default_width`\n\n    height: int or None\n        The height of the exported image in layout pixels. If the `scale`\n        property is 1.0, this will also be the height of the exported image\n        in physical pixels.\n\n        If not specified, will default to `plotly.io.config.default_height`\n\n    scale: int or float or None\n        The scale factor to use when exporting the figure. A scale factor\n        larger than 1.0 will increase the image resolution with respect\n        to the figure's layout pixel dimensions. Whereas as scale factor of\n        less than 1.0 will decrease the image resolution.\n\n        If not specified, will default to `plotly.io.config.default_scale`\n\n    validate: bool\n        True if the figure should be validated before being converted to\n        an image, False otherwise.\n\n    Returns\n    -------\n    bytes\n        The image data\n    "
    ensure_server()
    if format is None:
        format = config.default_format
    format = validate_coerce_format(format)
    if scale is None:
        scale = config.default_scale
    if width is None:
        width = config.default_width
    if height is None:
        height = config.default_height
    fig_dict = validate_coerce_fig_to_dict(fig, validate)
    try:
        response = request_image_with_retrying(figure=fig_dict, format=format, scale=scale, width=width, height=height)
    except OSError as err:
        status_str = repr(status)
        if config.server_url:
            raise ValueError('\nPlotly.py was unable to communicate with the orca server at {server_url}\n\nPlease check that the server is running and accessible.\n'.format(server_url=config.server_url))
        else:
            pid_exists = psutil.pid_exists(status.pid)
            if pid_exists:
                raise ValueError('\nFor some reason plotly.py was unable to communicate with the\nlocal orca server process, even though the server process seems to be running.\n\nPlease review the process and connection information below:\n\n{info}\n'.format(info=status_str))
            else:
                reset_status()
                raise ValueError('\nFor some reason the orca server process is no longer running.\n\nPlease review the process and connection information below:\n\n{info}\nplotly.py will attempt to start the local server process again the next time\nan image export operation is performed.\n'.format(info=status_str))
    if response.status_code == 200:
        return response.content
    else:
        err_message = '\nThe image request was rejected by the orca conversion utility\nwith the following error:\n   {status}: {msg}\n'.format(status=response.status_code, msg=response.content.decode('utf-8'))
        if response.status_code == 400 and isinstance(fig, dict) and (not validate):
            err_message += '\nTry setting the `validate` argument to True to check for errors in the\nfigure specification'
        elif response.status_code == 525:
            any_mapbox = any([trace.get('type', None) == 'scattermapbox' for trace in fig_dict.get('data', [])])
            if any_mapbox and config.mapbox_access_token is None:
                err_message += "\nExporting scattermapbox traces requires a mapbox access token.\nCreate a token in your mapbox account and then set it using:\n\n>>> plotly.io.orca.config.mapbox_access_token = 'pk.abc...'\n\nIf you would like this token to be applied automatically in\nfuture sessions, then save your orca configuration as follows:\n\n>>> plotly.io.orca.config.save()\n"
        elif response.status_code == 530 and format == 'eps':
            err_message += "\nExporting to EPS format requires the poppler library.  You can install\npoppler on MacOS or Linux with:\n\n    $ conda install poppler\n\nOr, you can install it on MacOS using homebrew with:\n\n    $ brew install poppler\n\nOr, you can install it on Linux using your distribution's package manager to\ninstall the 'poppler-utils' package.\n\nUnfortunately, we don't yet know of an easy way to install poppler on Windows.\n"
        raise ValueError(err_message)

def write_image(fig, file, format=None, scale=None, width=None, height=None, validate=True):
    if False:
        print('Hello World!')
    "\n    Convert a figure to a static image and write it to a file or writeable\n    object\n\n    Parameters\n    ----------\n    fig:\n        Figure object or dict representing a figure\n\n    file: str or writeable\n        A string representing a local file path or a writeable object\n        (e.g. a pathlib.Path object or an open file descriptor)\n\n    format: str or None\n        The desired image format. One of\n          - 'png'\n          - 'jpg' or 'jpeg'\n          - 'webp'\n          - 'svg'\n          - 'pdf'\n          - 'eps' (Requires the poppler library to be installed)\n\n        If not specified and `file` is a string then this will default to the\n        file extension. If not specified and `file` is not a string then this\n        will default to `plotly.io.config.default_format`\n\n    width: int or None\n        The width of the exported image in layout pixels. If the `scale`\n        property is 1.0, this will also be the width of the exported image\n        in physical pixels.\n\n        If not specified, will default to `plotly.io.config.default_width`\n\n    height: int or None\n        The height of the exported image in layout pixels. If the `scale`\n        property is 1.0, this will also be the height of the exported image\n        in physical pixels.\n\n        If not specified, will default to `plotly.io.config.default_height`\n\n    scale: int or float or None\n        The scale factor to use when exporting the figure. A scale factor\n        larger than 1.0 will increase the image resolution with respect\n        to the figure's layout pixel dimensions. Whereas as scale factor of\n        less than 1.0 will decrease the image resolution.\n\n        If not specified, will default to `plotly.io.config.default_scale`\n\n    validate: bool\n        True if the figure should be validated before being converted to\n        an image, False otherwise.\n\n    Returns\n    -------\n    None\n    "
    if isinstance(file, str):
        path = Path(file)
    elif isinstance(file, Path):
        path = file
    else:
        path = None
    if path is not None and format is None:
        ext = path.suffix
        if ext:
            format = ext.lstrip('.')
        else:
            raise ValueError("\nCannot infer image type from output path '{file}'.\nPlease add a file extension or specify the type using the format parameter.\nFor example:\n\n    >>> import plotly.io as pio\n    >>> pio.write_image(fig, file_path, format='png')\n".format(file=file))
    img_data = to_image(fig, format=format, scale=scale, width=width, height=height, validate=validate)
    if path is None:
        try:
            file.write(img_data)
            return
        except AttributeError:
            pass
        raise ValueError("\nThe 'file' argument '{file}' is not a string, pathlib.Path object, or file descriptor.\n".format(file=file))
    else:
        path.write_bytes(img_data)