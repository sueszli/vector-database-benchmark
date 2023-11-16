"""
Util functions for the NXOS modules.
"""
import collections
import http.client
import json
import logging
import os
import re
import socket
from collections.abc import Iterable
import salt.utils.http
from salt.exceptions import CommandExecutionError, NxosClientError, NxosError, NxosRequestNotSupported
from salt.utils.args import clean_kwargs
log = logging.getLogger(__name__)

class UHTTPConnection(http.client.HTTPConnection):
    """
    Subclass of Python library HTTPConnection that uses a unix-domain socket.
    """

    def __init__(self, path):
        if False:
            while True:
                i = 10
        http.client.HTTPConnection.__init__(self, 'localhost')
        self.path = path

    def connect(self):
        if False:
            while True:
                i = 10
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(self.path)
        self.sock = sock

class NxapiClient:
    """
    Class representing an NX-API client that connects over http(s) or
    unix domain socket (UDS).
    """
    NXAPI_UDS = '/tmp/nginx_local/nginx_1_be_nxapi.sock'
    NXAPI_REMOTE_URI_PATH = '/ins'
    NXAPI_UDS_URI_PATH = '/ins_local'
    NXAPI_VERSION = '1.0'

    def __init__(self, **nxos_kwargs):
        if False:
            return 10
        "\n        Initialize NxapiClient() connection object.  By default this connects\n        to the local unix domain socket (UDS).  If http(s) is required to\n        connect to a remote device then\n            nxos_kwargs['host'],\n            nxos_kwargs['username'],\n            nxos_kwargs['password'],\n            nxos_kwargs['transport'],\n            nxos_kwargs['port'],\n        parameters must be provided.\n        "
        self.nxargs = self._prepare_conn_args(clean_kwargs(**nxos_kwargs))
        if self.nxargs['connect_over_uds']:
            if not os.path.exists(self.NXAPI_UDS):
                raise NxosClientError('No host specified and no UDS found at {}\n'.format(self.NXAPI_UDS))
            log.info('Nxapi connection arguments: %s', self.nxargs)
            log.info('Connecting over unix domain socket')
            self.connection = UHTTPConnection(self.NXAPI_UDS)
        else:
            log.info('Nxapi connection arguments: %s', self.nxargs)
            log.info('Connecting over %s', self.nxargs['transport'])
            self.connection = salt.utils.http.query

    def _use_remote_connection(self, kwargs):
        if False:
            return 10
        '\n        Determine if connection is local or remote\n        '
        kwargs['host'] = kwargs.get('host')
        kwargs['username'] = kwargs.get('username')
        kwargs['password'] = kwargs.get('password')
        if kwargs['host'] is None or kwargs['username'] is None or kwargs['password'] is None:
            return False
        else:
            return True

    def _prepare_conn_args(self, kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set connection arguments for remote or local connection.\n        '
        kwargs['connect_over_uds'] = True
        kwargs['timeout'] = kwargs.get('timeout', 60)
        kwargs['cookie'] = kwargs.get('cookie', 'admin')
        if self._use_remote_connection(kwargs):
            kwargs['transport'] = kwargs.get('transport', 'https')
            if kwargs['transport'] == 'https':
                kwargs['port'] = kwargs.get('port', 443)
            else:
                kwargs['port'] = kwargs.get('port', 80)
            kwargs['verify'] = kwargs.get('verify', True)
            if isinstance(kwargs['verify'], bool):
                kwargs['verify_ssl'] = kwargs['verify']
            else:
                kwargs['ca_bundle'] = kwargs['verify']
            kwargs['connect_over_uds'] = False
        return kwargs

    def _build_request(self, type, commands):
        if False:
            while True:
                i = 10
        '\n        Build NX-API JSON request.\n        '
        request = {}
        headers = {'content-type': 'application/json'}
        if self.nxargs['connect_over_uds']:
            user = self.nxargs['cookie']
            headers['cookie'] = 'nxapi_auth=' + user + ':local'
            request['url'] = self.NXAPI_UDS_URI_PATH
        else:
            request['url'] = '{transport}://{host}:{port}{uri}'.format(transport=self.nxargs['transport'], host=self.nxargs['host'], port=self.nxargs['port'], uri=self.NXAPI_REMOTE_URI_PATH)
        if isinstance(commands, (list, set, tuple)):
            commands = ' ; '.join(commands)
        payload = {}
        payload['ins_api'] = collections.OrderedDict()
        payload['ins_api']['version'] = self.NXAPI_VERSION
        payload['ins_api']['type'] = type
        payload['ins_api']['chunk'] = '0'
        payload['ins_api']['sid'] = '1'
        payload['ins_api']['input'] = commands
        payload['ins_api']['output_format'] = 'json'
        request['headers'] = headers
        request['payload'] = json.dumps(payload)
        request['opts'] = {'http_request_timeout': self.nxargs['timeout']}
        log.info('request: %s', request)
        return request

    def request(self, type, command_list):
        if False:
            while True:
                i = 10
        '\n        Send NX-API JSON request to the NX-OS device.\n        '
        req = self._build_request(type, command_list)
        if self.nxargs['connect_over_uds']:
            self.connection.request('POST', req['url'], req['payload'], req['headers'])
            response = self.connection.getresponse()
        else:
            response = self.connection(req['url'], method='POST', opts=req['opts'], data=req['payload'], header_dict=req['headers'], decode=True, decode_type='json', **self.nxargs)
        return self.parse_response(response, command_list)

    def parse_response(self, response, command_list):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parse NX-API JSON response from the NX-OS device.\n        '
        if isinstance(response, Iterable) and 'status' in response:
            if int(response['status']) >= 500:
                raise NxosError('{}'.format(response))
            else:
                raise NxosError('NX-API Request Not Supported: {}'.format(response))
        if isinstance(response, Iterable):
            body = response['dict']
        else:
            body = response
        if self.nxargs['connect_over_uds']:
            body = json.loads(response.read().decode('utf-8'))
        output = body.get('ins_api')
        if output is None:
            raise NxosClientError('Unexpected JSON output\n{}'.format(body))
        if output.get('outputs'):
            output = output['outputs']
        if output.get('output'):
            output = output['output']
        result = []
        previous_commands = []
        if not isinstance(output, list):
            output = [output]
        if not isinstance(command_list, list):
            command_list = [command_list]
        if len(command_list) == 1 and ';' in command_list[0]:
            command_list = [cmd.strip() for cmd in command_list[0].split(';')]
        for (cmd_result, cmd) in zip(output, command_list):
            code = cmd_result.get('code')
            msg = cmd_result.get('msg')
            log.info('command %s:', cmd)
            log.info('PARSE_RESPONSE: %s %s', code, msg)
            if code == '400':
                raise CommandExecutionError({'rejected_input': cmd, 'code': code, 'message': msg, 'cli_error': cmd_result.get('clierror'), 'previous_commands': previous_commands})
            elif code == '413':
                raise NxosRequestNotSupported('Error 413: {}'.format(msg))
            elif code != '200':
                raise NxosError('Unknown Error: {}, Code: {}'.format(msg, code))
            else:
                previous_commands.append(cmd)
                result.append(cmd_result['body'])
        return result

def nxapi_request(commands, method='cli_show', **kwargs):
    if False:
        print('Hello World!')
    '\n    Send exec and config commands to the NX-OS device over NX-API.\n\n    commands\n        The exec or config commands to be sent.\n\n    method:\n        ``cli_show_ascii``: Return raw test or unstructured output.\n        ``cli_show``: Return structured output.\n        ``cli_conf``: Send configuration commands to the device.\n        Defaults to ``cli_show``.\n\n    transport: ``https``\n        Specifies the type of connection transport to use. Valid values for the\n        connection are ``http``, and  ``https``.\n\n    host: ``localhost``\n        The IP address or DNS host name of the device.\n\n    username: ``admin``\n        The username to pass to the device to authenticate the NX-API connection.\n\n    password\n        The password to pass to the device to authenticate the NX-API connection.\n\n    port\n        The TCP port of the endpoint for the NX-API connection. If this keyword is\n        not specified, the default value is automatically determined by the\n        transport type (``80`` for ``http``, or ``443`` for ``https``).\n\n    timeout: ``60``\n        Time in seconds to wait for the device to respond. Default: 60 seconds.\n\n    verify: ``True``\n        Either a boolean, in which case it controls whether we verify the NX-API\n        TLS certificate, or a string, in which case it must be a path to a CA bundle\n        to use. Defaults to ``True``.\n    '
    client = NxapiClient(**kwargs)
    return client.request(method, commands)

def ping(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Verify connection to the NX-OS device over UDS.\n    '
    return NxapiClient(**kwargs).nxargs['connect_over_uds']

def _parser(block):
    if False:
        print('Hello World!')
    return re.compile('^{block}\n(?:^[ \n].*$\n?)+'.format(block=block), re.MULTILINE)

def _parse_software(data):
    if False:
        print('Hello World!')
    '\n    Internal helper function to parse sotware grain information.\n    '
    ret = {'software': {}}
    software = _parser('Software').search(data).group(0)
    matcher = re.compile('^  ([^:]+): *([^\n]+)', re.MULTILINE)
    for line in matcher.finditer(software):
        (key, val) = line.groups()
        ret['software'][key] = val
    return ret['software']

def _parse_hardware(data):
    if False:
        print('Hello World!')
    '\n    Internal helper function to parse hardware grain information.\n    '
    ret = {'hardware': {}}
    hardware = _parser('Hardware').search(data).group(0)
    matcher = re.compile('^  ([^:\n]+): *([^\n]+)', re.MULTILINE)
    for line in matcher.finditer(hardware):
        (key, val) = line.groups()
        ret['hardware'][key] = val
    return ret['hardware']

def _parse_plugins(data):
    if False:
        i = 10
        return i + 15
    '\n    Internal helper function to parse plugin grain information.\n    '
    ret = {'plugins': []}
    plugins = _parser('plugin').search(data).group(0)
    matcher = re.compile('^  (?:([^,]+), )+([^\n]+)', re.MULTILINE)
    for line in matcher.finditer(plugins):
        ret['plugins'].extend(line.groups())
    return ret['plugins']

def version_info():
    if False:
        while True:
            i = 10
    client = NxapiClient()
    return client.request('cli_show_ascii', 'show version')[0]

def system_info(data):
    if False:
        return 10
    "\n    Helper method to return parsed system_info\n    from the 'show version' command.\n    "
    if not data:
        return {}
    info = {'software': _parse_software(data), 'hardware': _parse_hardware(data), 'plugins': _parse_plugins(data)}
    return {'nxos': info}