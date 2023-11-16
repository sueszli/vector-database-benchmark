"""
Util functions for the NXOS API modules.
"""
import json
import logging
import salt.utils.http
from salt.exceptions import SaltException
from salt.utils.args import clean_kwargs
log = logging.getLogger(__name__)
RPC_INIT_KWARGS = ['transport', 'host', 'username', 'password', 'port', 'timeout', 'verify', 'rpc_version']

def _prepare_connection(**nxos_api_kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Prepare the connection with the remote network device, and clean up the key\n    value pairs, removing the args used for the connection init.\n    '
    nxos_api_kwargs = clean_kwargs(**nxos_api_kwargs)
    init_kwargs = {}
    for (karg, warg) in nxos_api_kwargs.items():
        if karg in RPC_INIT_KWARGS:
            init_kwargs[karg] = warg
    if 'host' not in init_kwargs:
        init_kwargs['host'] = 'localhost'
    if 'transport' not in init_kwargs:
        init_kwargs['transport'] = 'https'
    if 'port' not in init_kwargs:
        init_kwargs['port'] = 80 if init_kwargs['transport'] == 'http' else 443
    verify = init_kwargs.get('verify', True)
    if isinstance(verify, bool):
        init_kwargs['verify_ssl'] = verify
    else:
        init_kwargs['ca_bundle'] = verify
    if 'rpc_version' not in init_kwargs:
        init_kwargs['rpc_version'] = '2.0'
    if 'timeout' not in init_kwargs:
        init_kwargs['timeout'] = 60
    return init_kwargs

def rpc(commands, method='cli', **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Execute an arbitrary RPC request via the Nexus API.\n\n    commands\n        The commands to be executed.\n\n    method: ``cli``\n        The type of the response, i.e., raw text (``cli_ascii``) or structured\n        document (``cli``). Defaults to ``cli`` (structured data).\n\n    transport: ``https``\n        Specifies the type of connection transport to use. Valid values for the\n        connection are ``http``, and  ``https``.\n\n    host: ``localhost``\n        The IP address or DNS host name of the connection device.\n\n    username: ``admin``\n        The username to pass to the device to authenticate the NX-API connection.\n\n    password\n        The password to pass to the device to authenticate the NX-API connection.\n\n    port\n        The TCP port of the endpoint for the NX-API connection. If this keyword is\n        not specified, the default value is automatically determined by the\n        transport type (``80`` for ``http``, or ``443`` for ``https``).\n\n    timeout: ``60``\n        Time in seconds to wait for the device to respond. Default: 60 seconds.\n\n    verify: ``True``\n        Either a boolean, in which case it controls whether we verify the NX-API\n        TLS certificate, or a string, in which case it must be a path to a CA bundle\n        to use. Defaults to ``True``.\n    '
    init_args = _prepare_connection(**kwargs)
    log.error('These are the init args:')
    log.error(init_args)
    url = '{transport}://{host}:{port}/ins'.format(transport=init_args['transport'], host=init_args['host'], port=init_args['port'])
    headers = {'content-type': 'application/json-rpc'}
    payload = []
    if not isinstance(commands, (list, tuple)):
        commands = [commands]
    for (index, command) in enumerate(commands):
        payload.append({'jsonrpc': init_args['rpc_version'], 'method': method, 'params': {'cmd': command, 'version': 1}, 'id': index + 1})
    opts = {'http_request_timeout': init_args['timeout']}
    response = salt.utils.http.query(url, method='POST', opts=opts, data=json.dumps(payload), header_dict=headers, decode=True, decode_type='json', **init_args)
    if 'error' in response:
        raise SaltException(response['error'])
    response_list = response['dict']
    if isinstance(response_list, dict):
        response_list = [response_list]
    for (index, command) in enumerate(commands):
        response_list[index]['command'] = command
    return response_list