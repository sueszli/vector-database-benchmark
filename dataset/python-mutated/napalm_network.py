"""
NAPALM Network
==============

Basic methods for interaction with the network device through the virtual proxy 'napalm'.

:codeauthor: Mircea Ulinic <ping@mirceaulinic.net> & Jerome Fleury <jf@cloudflare.com>
:maturity:   new
:depends:    napalm
:platform:   unix

Dependencies
------------

- :mod:`napalm proxy minion <salt.proxy.napalm>`

.. versionadded:: 2016.11.0
.. versionchanged:: 2017.7.0
"""
import datetime
import logging
import time
import salt.utils.files
import salt.utils.napalm
import salt.utils.templates
import salt.utils.versions
log = logging.getLogger(__name__)
try:
    import jxmlease
    HAS_JXMLEASE = True
except ImportError:
    HAS_JXMLEASE = False
__virtualname__ = 'net'
__proxyenabled__ = ['*']
__virtual_aliases__ = ('napalm_net',)

def __virtual__():
    if False:
        print('Hello World!')
    '\n    NAPALM library must be installed for this module to work and run in a (proxy) minion.\n    '
    return salt.utils.napalm.virtual(__opts__, __virtualname__, __file__)

def _filter_list(input_list, search_key, search_value):
    if False:
        i = 10
        return i + 15
    '\n    Filters a list of dictionary by a set of key-value pair.\n\n    :param input_list:   is a list of dictionaries\n    :param search_key:   is the key we are looking for\n    :param search_value: is the value we are looking for the key specified in search_key\n    :return:             filered list of dictionaries\n    '
    output_list = list()
    for dictionary in input_list:
        if dictionary.get(search_key) == search_value:
            output_list.append(dictionary)
    return output_list

def _filter_dict(input_dict, search_key, search_value):
    if False:
        for i in range(10):
            print('nop')
    '\n    Filters a dictionary of dictionaries by a key-value pair.\n\n    :param input_dict:    is a dictionary whose values are lists of dictionaries\n    :param search_key:    is the key in the leaf dictionaries\n    :param search_values: is the value in the leaf dictionaries\n    :return:              filtered dictionary\n    '
    output_dict = dict()
    for (key, key_list) in input_dict.items():
        key_list_filtered = _filter_list(key_list, search_key, search_value)
        if key_list_filtered:
            output_dict[key] = key_list_filtered
    return output_dict

def _safe_commit_config(loaded_result, napalm_device):
    if False:
        return 10
    _commit = commit(inherit_napalm_device=napalm_device)
    if not _commit.get('result', False):
        loaded_result['comment'] += _commit['comment'] if _commit.get('comment') else 'Unable to commit.'
        loaded_result['result'] = False
        discarded = _safe_dicard_config(loaded_result, napalm_device)
        if not discarded['result']:
            return loaded_result
    return _commit

def _safe_dicard_config(loaded_result, napalm_device):
    if False:
        return 10
    log.debug('Discarding the config')
    log.debug(loaded_result)
    _discarded = discard_config(inherit_napalm_device=napalm_device)
    if not _discarded.get('result', False):
        loaded_result['comment'] += _discarded['comment'] if _discarded.get('comment') else 'Unable to discard config.'
        loaded_result['result'] = False
        _explicit_close(napalm_device)
        __context__['retcode'] = 1
        return loaded_result
    return _discarded

def _explicit_close(napalm_device):
    if False:
        for i in range(10):
            print('nop')
    '\n    Will explicily close the config session with the network device,\n    when running in a now-always-alive proxy minion or regular minion.\n    This helper must be used in configuration-related functions,\n    as the session is preserved and not closed before making any changes.\n    '
    if salt.utils.napalm.not_always_alive(__opts__):
        try:
            napalm_device['DRIVER'].close()
        except Exception as err:
            log.error('Unable to close the temp connection with the device:')
            log.error(err)
            log.error('Please report.')

def _config_logic(napalm_device, loaded_result, test=False, debug=False, replace=False, commit_config=True, loaded_config=None, commit_in=None, commit_at=None, revert_in=None, revert_at=None, commit_jid=None, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Builds the config logic for `load_config` and `load_template` functions.\n    '
    current_jid = kwargs.get('__pub_jid')
    if not current_jid:
        current_jid = '{:%Y%m%d%H%M%S%f}'.format(datetime.datetime.now())
    loaded_result['already_configured'] = False
    loaded_result['loaded_config'] = ''
    if debug:
        loaded_result['loaded_config'] = loaded_config
    _compare = compare_config(inherit_napalm_device=napalm_device)
    if _compare.get('result', False):
        loaded_result['diff'] = _compare.get('out')
        loaded_result.pop('out', '')
    else:
        loaded_result['diff'] = None
        loaded_result['result'] = False
        loaded_result['comment'] = _compare.get('comment')
        __context__['retcode'] = 1
        return loaded_result
    _loaded_res = loaded_result.get('result', False)
    if not _loaded_res or test:
        if loaded_result['comment']:
            loaded_result['comment'] += '\n'
        if not loaded_result.get('diff', ''):
            loaded_result['already_configured'] = True
        discarded = _safe_dicard_config(loaded_result, napalm_device)
        if not discarded['result']:
            return loaded_result
        loaded_result['comment'] += 'Configuration discarded.'
        _explicit_close(napalm_device)
        if not loaded_result['result']:
            __context__['retcode'] = 1
        return loaded_result
    if not test and commit_config:
        if commit_jid:
            log.info('Committing the JID: %s', str(commit_jid))
            removed = cancel_commit(commit_jid)
            log.debug('Cleaned up the commit from the schedule')
            log.debug(removed['comment'])
        if loaded_result.get('diff', ''):
            if commit_in or commit_at:
                commit_time = __utils__['timeutil.get_time_at'](time_in=commit_in, time_at=commit_in)
                scheduled_job_name = '__napalm_commit_{}'.format(current_jid)
                temp_file = salt.utils.files.mkstemp()
                with salt.utils.files.fopen(temp_file, 'w') as fp_:
                    fp_.write(loaded_config)
                scheduled = __salt__['schedule.add'](scheduled_job_name, function='net.load_config', job_kwargs={'filename': temp_file, 'commit_jid': current_jid, 'replace': replace}, once=commit_time)
                log.debug('Scheduling job')
                log.debug(scheduled)
                saved = __salt__['schedule.save']()
                discarded = _safe_dicard_config(loaded_result, napalm_device)
                if not discarded['result']:
                    discarded['comment'] += 'Scheduled the job to be executed at {schedule_ts}, but was unable to discard the config: \n'.format(schedule_ts=commit_time)
                    return discarded
                loaded_result['comment'] = 'Changes discarded for now, and scheduled commit at: {schedule_ts}.\nThe commit ID is: {current_jid}.\nTo discard this commit, you can execute: \n\nsalt {min_id} net.cancel_commit {current_jid}'.format(schedule_ts=commit_time, min_id=__opts__['id'], current_jid=current_jid)
                loaded_result['commit_id'] = current_jid
                return loaded_result
            log.debug('About to commit:')
            log.debug(loaded_result['diff'])
            if revert_in or revert_at:
                revert_time = __utils__['timeutil.get_time_at'](time_in=revert_in, time_at=revert_at)
                if __grains__['os'] == 'junos':
                    if not HAS_JXMLEASE:
                        loaded_result['comment'] = 'This feature requires the library jxmlease to be installed.\nTo install, please execute: ``pip install jxmlease``.'
                        loaded_result['result'] = False
                        return loaded_result
                    timestamp_at = __utils__['timeutil.get_timestamp_at'](time_in=revert_in, time_at=revert_at)
                    minutes = int((timestamp_at - time.time()) / 60)
                    _comm = __salt__['napalm.junos_commit'](confirm=minutes)
                    if not _comm['out']:
                        loaded_result['comment'] = 'Unable to commit confirm: {}'.format(_comm['message'])
                        loaded_result['result'] = False
                        discarded = _safe_dicard_config(loaded_result, napalm_device)
                        if not discarded['result']:
                            return loaded_result
                else:
                    temp_file = salt.utils.files.mkstemp()
                    running_config = __salt__['net.config'](source='running')['out']['running']
                    with salt.utils.files.fopen(temp_file, 'w') as fp_:
                        fp_.write(running_config)
                    committed = _safe_commit_config(loaded_result, napalm_device)
                    if not committed['result']:
                        return loaded_result
                    scheduled_job_name = '__napalm_commit_{}'.format(current_jid)
                    scheduled = __salt__['schedule.add'](scheduled_job_name, function='net.load_config', job_kwargs={'filename': temp_file, 'commit_jid': current_jid, 'replace': True}, once=revert_time)
                    log.debug('Scheduling commit confirmed')
                    log.debug(scheduled)
                    saved = __salt__['schedule.save']()
                loaded_result['comment'] = 'The commit ID is: {current_jid}.\nThis commit will be reverted at: {schedule_ts}, unless confirmed.\nTo confirm the commit and avoid reverting, you can execute:\n\nsalt {min_id} net.confirm_commit {current_jid}'.format(schedule_ts=revert_time, min_id=__opts__['id'], current_jid=current_jid)
                loaded_result['commit_id'] = current_jid
                return loaded_result
            committed = _safe_commit_config(loaded_result, napalm_device)
            if not committed['result']:
                return loaded_result
        else:
            discarded = _safe_dicard_config(loaded_result, napalm_device)
            if not discarded['result']:
                return loaded_result
            loaded_result['already_configured'] = True
            loaded_result['comment'] = 'Already configured.'
    _explicit_close(napalm_device)
    if not loaded_result['result']:
        __context__['retcode'] = 1
    return loaded_result

@salt.utils.napalm.proxy_napalm_wrap
def connected(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Specifies if the connection to the device succeeded.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' net.connected\n    "
    return {'out': napalm_device.get('UP', False)}

@salt.utils.napalm.proxy_napalm_wrap
def facts(**kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Returns characteristics of the network device.\n    :return: a dictionary with the following keys:\n\n        * uptime - Uptime of the device in seconds.\n        * vendor - Manufacturer of the device.\n        * model - Device model.\n        * hostname - Hostname of the device\n        * fqdn - Fqdn of the device\n        * os_version - String with the OS version running on the device.\n        * serial_number - Serial number of the device\n        * interface_list - List of the interfaces of the device\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' net.facts\n\n    Example output:\n\n    .. code-block:: python\n\n        {\n            'os_version': '13.3R6.5',\n            'uptime': 10117140,\n            'interface_list': [\n                'lc-0/0/0',\n                'pfe-0/0/0',\n                'pfh-0/0/0',\n                'xe-0/0/0',\n                'xe-0/0/1',\n                'xe-0/0/2',\n                'xe-0/0/3',\n                'gr-0/0/10',\n                'ip-0/0/10'\n            ],\n            'vendor': 'Juniper',\n            'serial_number': 'JN131356FBFA',\n            'model': 'MX480',\n            'hostname': 're0.edge05.syd01',\n            'fqdn': 're0.edge05.syd01'\n        }\n    "
    return salt.utils.napalm.call(napalm_device, 'get_facts', **{})

@salt.utils.napalm.proxy_napalm_wrap
def environment(**kwargs):
    if False:
        print('Hello World!')
    "\n    Returns the environment of the device.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' net.environment\n\n\n    Example output:\n\n    .. code-block:: python\n\n        {\n            'fans': {\n                'Bottom Rear Fan': {\n                    'status': True\n                },\n                'Bottom Middle Fan': {\n                    'status': True\n                },\n                'Top Middle Fan': {\n                    'status': True\n                },\n                'Bottom Front Fan': {\n                    'status': True\n                },\n                'Top Front Fan': {\n                    'status': True\n                },\n                'Top Rear Fan': {\n                    'status': True\n                }\n            },\n            'memory': {\n                'available_ram': 16349,\n                'used_ram': 4934\n            },\n            'temperature': {\n               'FPC 0 Exhaust A': {\n                    'is_alert': False,\n                    'temperature': 35.0,\n                    'is_critical': False\n                }\n            },\n            'cpu': {\n                '1': {\n                    '%usage': 19.0\n                },\n                '0': {\n                    '%usage': 35.0\n                }\n            }\n        }\n    "
    return salt.utils.napalm.call(napalm_device, 'get_environment', **{})

@salt.utils.napalm.proxy_napalm_wrap
def cli(*commands, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Returns a dictionary with the raw output of all commands passed as arguments.\n\n    commands\n        List of commands to be executed on the device.\n\n    textfsm_parse: ``False``\n        Try parsing the outputs using the TextFSM templates.\n\n        .. versionadded:: 2018.3.0\n\n        .. note::\n            This option can be also specified in the minion configuration\n            file or pillar as ``napalm_cli_textfsm_parse``.\n\n    textfsm_path\n        The path where the TextFSM templates can be found. This option implies\n        the usage of the TextFSM index file.\n        ``textfsm_path`` can be either absolute path on the server,\n        either specified using the following URL mschemes: ``file://``,\n        ``salt://``, ``http://``, ``https://``, ``ftp://``,\n        ``s3://``, ``swift://``.\n\n        .. versionadded:: 2018.3.0\n\n        .. note::\n            This needs to be a directory with a flat structure, having an\n            index file (whose name can be specified using the ``index_file`` option)\n            and a number of TextFSM templates.\n\n        .. note::\n            This option can be also specified in the minion configuration\n            file or pillar as ``textfsm_path``.\n\n    textfsm_template\n        The path to a certain the TextFSM template.\n        This can be specified using the absolute path\n        to the file, or using one of the following URL schemes:\n\n        - ``salt://``, to fetch the template from the Salt fileserver.\n        - ``http://`` or ``https://``\n        - ``ftp://``\n        - ``s3://``\n        - ``swift://``\n\n        .. versionadded:: 2018.3.0\n\n    textfsm_template_dict\n        A dictionary with the mapping between a command\n        and the corresponding TextFSM path to use to extract the data.\n        The TextFSM paths can be specified as in ``textfsm_template``.\n\n        .. versionadded:: 2018.3.0\n\n        .. note::\n            This option can be also specified in the minion configuration\n            file or pillar as ``napalm_cli_textfsm_template_dict``.\n\n    platform_grain_name: ``os``\n        The name of the grain used to identify the platform name\n        in the TextFSM index file. Default: ``os``.\n\n        .. versionadded:: 2018.3.0\n\n        .. note::\n            This option can be also specified in the minion configuration\n            file or pillar as ``textfsm_platform_grain``.\n\n    platform_column_name: ``Platform``\n        The column name used to identify the platform,\n        exactly as specified in the TextFSM index file.\n        Default: ``Platform``.\n\n        .. versionadded:: 2018.3.0\n\n        .. note::\n            This is field is case sensitive, make sure\n            to assign the correct value to this option,\n            exactly as defined in the index file.\n\n        .. note::\n            This option can be also specified in the minion configuration\n            file or pillar as ``textfsm_platform_column_name``.\n\n    index_file: ``index``\n        The name of the TextFSM index file, under the ``textfsm_path``. Default: ``index``.\n\n        .. versionadded:: 2018.3.0\n\n        .. note::\n            This option can be also specified in the minion configuration\n            file or pillar as ``textfsm_index_file``.\n\n    saltenv: ``base``\n        Salt fileserver environment from which to retrieve the file.\n        Ignored if ``textfsm_path`` is not a ``salt://`` URL.\n\n        .. versionadded:: 2018.3.0\n\n    include_empty: ``False``\n        Include empty files under the ``textfsm_path``.\n\n        .. versionadded:: 2018.3.0\n\n    include_pat\n        Glob or regex to narrow down the files cached from the given path.\n        If matching with a regex, the regex must be prefixed with ``E@``,\n        otherwise the expression will be interpreted as a glob.\n\n        .. versionadded:: 2018.3.0\n\n    exclude_pat\n        Glob or regex to exclude certain files from being cached from the given path.\n        If matching with a regex, the regex must be prefixed with ``E@``,\n        otherwise the expression will be interpreted as a glob.\n\n        .. versionadded:: 2018.3.0\n\n        .. note::\n            If used with ``include_pat``, files matching this pattern will be\n            excluded from the subset of files defined by ``include_pat``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' net.cli "show version" "show chassis fan"\n\n    CLI Example with TextFSM template:\n\n    .. code-block:: bash\n\n        salt \'*\' net.cli textfsm_parse=True textfsm_path=salt://textfsm/\n\n    Example output:\n\n    .. code-block:: python\n\n        {\n            \'show version and haiku\':  \'Hostname: re0.edge01.arn01\n                                          Model: mx480\n                                          Junos: 13.3R6.5\n                                            Help me, Obi-Wan\n                                            I just saw Episode Two\n                                            You\'re my only hope\n                                         \',\n            \'show chassis fan\' :   \'Item                      Status   RPM     Measurement\n                                      Top Rear Fan              OK       3840    Spinning at intermediate-speed\n                                      Bottom Rear Fan           OK       3840    Spinning at intermediate-speed\n                                      Top Middle Fan            OK       3900    Spinning at intermediate-speed\n                                      Bottom Middle Fan         OK       3840    Spinning at intermediate-speed\n                                      Top Front Fan             OK       3810    Spinning at intermediate-speed\n                                      Bottom Front Fan          OK       3840    Spinning at intermediate-speed\n                                     \'\n        }\n\n    Example output with TextFSM parsing:\n\n    .. code-block:: json\n\n        {\n          "comment": "",\n          "result": true,\n          "out": {\n            "sh ver": [\n              {\n                "kernel": "9.1S3.5",\n                "documentation": "9.1S3.5",\n                "boot": "9.1S3.5",\n                "crypto": "9.1S3.5",\n                "chassis": "",\n                "routing": "9.1S3.5",\n                "base": "9.1S3.5",\n                "model": "mx960"\n              }\n            ]\n          }\n        }\n    '
    raw_cli_outputs = salt.utils.napalm.call(napalm_device, 'cli', **{'commands': list(commands)})
    if not raw_cli_outputs['result']:
        return raw_cli_outputs
    textfsm_parse = kwargs.get('textfsm_parse') or __opts__.get('napalm_cli_textfsm_parse') or __pillar__.get('napalm_cli_textfsm_parse', False)
    if not textfsm_parse:
        log.debug('No TextFSM parsing requested.')
        return raw_cli_outputs
    if 'textfsm.extract' not in __salt__ or 'textfsm.index' not in __salt__:
        raw_cli_outputs['comment'] += 'Unable to process: is TextFSM installed?'
        log.error(raw_cli_outputs['comment'])
        return raw_cli_outputs
    textfsm_template = kwargs.get('textfsm_template')
    log.debug('textfsm_template: %s', textfsm_template)
    textfsm_path = kwargs.get('textfsm_path') or __opts__.get('textfsm_path') or __pillar__.get('textfsm_path')
    log.debug('textfsm_path: %s', textfsm_path)
    textfsm_template_dict = kwargs.get('textfsm_template_dict') or __opts__.get('napalm_cli_textfsm_template_dict') or __pillar__.get('napalm_cli_textfsm_template_dict', {})
    log.debug('TextFSM command-template mapping: %s', textfsm_template_dict)
    index_file = kwargs.get('index_file') or __opts__.get('textfsm_index_file') or __pillar__.get('textfsm_index_file')
    log.debug('index_file: %s', index_file)
    platform_grain_name = kwargs.get('platform_grain_name') or __opts__.get('textfsm_platform_grain') or __pillar__.get('textfsm_platform_grain', 'os')
    log.debug('platform_grain_name: %s', platform_grain_name)
    platform_column_name = kwargs.get('platform_column_name') or __opts__.get('textfsm_platform_column_name') or __pillar__.get('textfsm_platform_column_name', 'Platform')
    log.debug('platform_column_name: %s', platform_column_name)
    saltenv = kwargs.get('saltenv', 'base')
    include_empty = kwargs.get('include_empty', False)
    include_pat = kwargs.get('include_pat')
    exclude_pat = kwargs.get('exclude_pat')
    processed_cli_outputs = {'comment': raw_cli_outputs.get('comment', ''), 'result': raw_cli_outputs['result'], 'out': {}}
    log.debug('Starting to analyse the raw outputs')
    for command in list(commands):
        command_output = raw_cli_outputs['out'][command]
        log.debug('Output from command: %s', command)
        log.debug(command_output)
        processed_command_output = None
        if textfsm_path:
            log.debug('Using the templates under %s', textfsm_path)
            processed_cli_output = __salt__['textfsm.index'](command, platform_grain_name=platform_grain_name, platform_column_name=platform_column_name, output=command_output.strip(), textfsm_path=textfsm_path, saltenv=saltenv, include_empty=include_empty, include_pat=include_pat, exclude_pat=exclude_pat)
            log.debug('Processed CLI output:')
            log.debug(processed_cli_output)
            if not processed_cli_output['result']:
                log.debug('Apparently this did not work, returning the raw output')
                processed_command_output = command_output
                processed_cli_outputs['comment'] += '\nUnable to process the output from {}: {}.'.format(command, processed_cli_output['comment'])
                log.error(processed_cli_outputs['comment'])
            elif processed_cli_output['out']:
                log.debug('All good, %s has a nice output!', command)
                processed_command_output = processed_cli_output['out']
            else:
                comment = '\nProcessing "{}" didn\'t fail, but didn\'t return anything either. Dumping raw.'.format(command)
                processed_cli_outputs['comment'] += comment
                log.error(comment)
                processed_command_output = command_output
        elif textfsm_template or command in textfsm_template_dict:
            if command in textfsm_template_dict:
                textfsm_template = textfsm_template_dict[command]
            log.debug('Using %s to process the command: %s', textfsm_template, command)
            processed_cli_output = __salt__['textfsm.extract'](textfsm_template, raw_text=command_output, saltenv=saltenv)
            log.debug('Processed CLI output:')
            log.debug(processed_cli_output)
            if not processed_cli_output['result']:
                log.debug('Apparently this did not work, returning the raw output')
                processed_command_output = command_output
                processed_cli_outputs['comment'] += '\nUnable to process the output from {}: {}'.format(command, processed_cli_output['comment'])
                log.error(processed_cli_outputs['comment'])
            elif processed_cli_output['out']:
                log.debug('All good, %s has a nice output!', command)
                processed_command_output = processed_cli_output['out']
            else:
                log.debug('Processing %s did not fail, but did not return anything either. Dumping raw.', command)
                processed_command_output = command_output
        else:
            log.error('No TextFSM template specified, or no TextFSM path defined')
            processed_command_output = command_output
            processed_cli_outputs['comment'] += '\nUnable to process the output from {}.'.format(command)
        processed_cli_outputs['out'][command] = processed_command_output
    processed_cli_outputs['comment'] = processed_cli_outputs['comment'].strip()
    return processed_cli_outputs

@salt.utils.napalm.proxy_napalm_wrap
def traceroute(destination, source=None, ttl=None, timeout=None, vrf=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Calls the method traceroute from the NAPALM driver object and returns a dictionary with the result of the traceroute\n    command executed on the device.\n\n    destination\n        Hostname or address of remote host\n\n    source\n        Source address to use in outgoing traceroute packets\n\n    ttl\n        IP maximum time-to-live value (or IPv6 maximum hop-limit value)\n\n    timeout\n        Number of seconds to wait for response (seconds)\n\n    vrf\n        VRF (routing instance) for traceroute attempt\n\n        .. versionadded:: 2016.11.4\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' net.traceroute 8.8.8.8\n        salt '*' net.traceroute 8.8.8.8 source=127.0.0.1 ttl=5 timeout=1\n    "
    return salt.utils.napalm.call(napalm_device, 'traceroute', **{'destination': destination, 'source': source, 'ttl': ttl, 'timeout': timeout, 'vrf': vrf})

@salt.utils.napalm.proxy_napalm_wrap
def ping(destination, source=None, ttl=None, timeout=None, size=None, count=None, vrf=None, **kwargs):
    if False:
        print('Hello World!')
    "\n    Executes a ping on the network device and returns a dictionary as a result.\n\n    destination\n        Hostname or IP address of remote host\n\n    source\n        Source address of echo request\n\n    ttl\n        IP time-to-live value (IPv6 hop-limit value) (1..255 hops)\n\n    timeout\n        Maximum wait time after sending final packet (seconds)\n\n    size\n        Size of request packets (0..65468 bytes)\n\n    count\n        Number of ping requests to send (1..2000000000 packets)\n\n    vrf\n        VRF (routing instance) for ping attempt\n\n        .. versionadded:: 2016.11.4\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' net.ping 8.8.8.8\n        salt '*' net.ping 8.8.8.8 ttl=3 size=65468\n        salt '*' net.ping 8.8.8.8 source=127.0.0.1 timeout=1 count=100\n    "
    return salt.utils.napalm.call(napalm_device, 'ping', **{'destination': destination, 'source': source, 'ttl': ttl, 'timeout': timeout, 'size': size, 'count': count, 'vrf': vrf})

@salt.utils.napalm.proxy_napalm_wrap
def arp(interface='', ipaddr='', macaddr='', **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    NAPALM returns a list of dictionaries with details of the ARP entries.\n\n    :param interface: interface name to filter on\n    :param ipaddr: IP address to filter on\n    :param macaddr: MAC address to filter on\n    :return: List of the entries in the ARP table\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' net.arp\n        salt '*' net.arp macaddr='5c:5e:ab:da:3c:f0'\n\n    Example output:\n\n    .. code-block:: python\n\n        [\n            {\n                'interface' : 'MgmtEth0/RSP0/CPU0/0',\n                'mac'       : '5c:5e:ab:da:3c:f0',\n                'ip'        : '172.17.17.1',\n                'age'       : 1454496274.84\n            },\n            {\n                'interface': 'MgmtEth0/RSP0/CPU0/0',\n                'mac'       : '66:0e:94:96:e0:ff',\n                'ip'        : '172.17.17.2',\n                'age'       : 1435641582.49\n            }\n        ]\n    "
    proxy_output = salt.utils.napalm.call(napalm_device, 'get_arp_table', **{})
    if not proxy_output.get('result'):
        return proxy_output
    arp_table = proxy_output.get('out')
    if interface:
        arp_table = _filter_list(arp_table, 'interface', interface)
    if ipaddr:
        arp_table = _filter_list(arp_table, 'ip', ipaddr)
    if macaddr:
        arp_table = _filter_list(arp_table, 'mac', macaddr)
    proxy_output.update({'out': arp_table})
    return proxy_output

@salt.utils.napalm.proxy_napalm_wrap
def ipaddrs(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns IP addresses configured on the device.\n\n    :return:   A dictionary with the IPv4 and IPv6 addresses of the interfaces.\n        Returns all configured IP addresses on all interfaces as a dictionary\n        of dictionaries.  Keys of the main dictionary represent the name of the\n        interface.  Values of the main dictionary represent are dictionaries\n        that may consist of two keys 'ipv4' and 'ipv6' (one, both or none)\n        which are themselvs dictionaries with the IP addresses as keys.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' net.ipaddrs\n\n    Example output:\n\n    .. code-block:: python\n\n        {\n            'FastEthernet8': {\n                'ipv4': {\n                    '10.66.43.169': {\n                        'prefix_length': 22\n                    }\n                }\n            },\n            'Loopback555': {\n                'ipv4': {\n                    '192.168.1.1': {\n                        'prefix_length': 24\n                    }\n                },\n                'ipv6': {\n                    '1::1': {\n                        'prefix_length': 64\n                    },\n                    '2001:DB8:1::1': {\n                        'prefix_length': 64\n                    },\n                    'FE80::3': {\n                        'prefix_length': 'N/A'\n                    }\n                }\n            }\n        }\n    "
    return salt.utils.napalm.call(napalm_device, 'get_interfaces_ip', **{})

@salt.utils.napalm.proxy_napalm_wrap
def interfaces(**kwargs):
    if False:
        while True:
            i = 10
    "\n    Returns details of the interfaces on the device.\n\n    :return: Returns a dictionary of dictionaries. The keys for the first\n        dictionary will be the interfaces in the devices.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' net.interfaces\n\n    Example output:\n\n    .. code-block:: python\n\n        {\n            'Management1': {\n                'is_up': False,\n                'is_enabled': False,\n                'description': '',\n                'last_flapped': -1,\n                'speed': 1000,\n                'mac_address': 'dead:beef:dead',\n            },\n            'Ethernet1':{\n                'is_up': True,\n                'is_enabled': True,\n                'description': 'foo',\n                'last_flapped': 1429978575.1554043,\n                'speed': 1000,\n                'mac_address': 'beef:dead:beef',\n            }\n        }\n    "
    return salt.utils.napalm.call(napalm_device, 'get_interfaces', **{})

@salt.utils.napalm.proxy_napalm_wrap
def lldp(interface='', **kwargs):
    if False:
        while True:
            i = 10
    "\n    Returns a detailed view of the LLDP neighbors.\n\n    :param interface: interface name to filter on\n\n    :return:          A dictionary with the LLDL neighbors. The keys are the\n        interfaces with LLDP activated on.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' net.lldp\n        salt '*' net.lldp interface='TenGigE0/0/0/8'\n\n    Example output:\n\n    .. code-block:: python\n\n        {\n            'TenGigE0/0/0/8': [\n                {\n                    'parent_interface': 'Bundle-Ether8',\n                    'interface_description': 'TenGigE0/0/0/8',\n                    'remote_chassis_id': '8c60.4f69.e96c',\n                    'remote_system_name': 'switch',\n                    'remote_port': 'Eth2/2/1',\n                    'remote_port_description': 'Ethernet2/2/1',\n                    'remote_system_description': 'Cisco Nexus Operating System (NX-OS) Software 7.1(0)N1(1a)\n                          TAC support: http://www.cisco.com/tac\n                          Copyright (c) 2002-2015, Cisco Systems, Inc. All rights reserved.',\n                    'remote_system_capab': 'B, R',\n                    'remote_system_enable_capab': 'B'\n                }\n            ]\n        }\n    "
    proxy_output = salt.utils.napalm.call(napalm_device, 'get_lldp_neighbors_detail', **{})
    if not proxy_output.get('result'):
        return proxy_output
    lldp_neighbors = proxy_output.get('out')
    if interface:
        lldp_neighbors = {interface: lldp_neighbors.get(interface)}
    proxy_output.update({'out': lldp_neighbors})
    return proxy_output

@salt.utils.napalm.proxy_napalm_wrap
def mac(address='', interface='', vlan=0, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Returns the MAC Address Table on the device.\n\n    :param address:   MAC address to filter on\n    :param interface: Interface name to filter on\n    :param vlan:      VLAN identifier\n    :return:          A list of dictionaries representing the entries in the MAC Address Table\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' net.mac\n        salt '*' net.mac vlan=10\n\n    Example output:\n\n    .. code-block:: python\n\n        [\n            {\n                'mac'       : '00:1c:58:29:4a:71',\n                'interface' : 'xe-3/0/2',\n                'static'    : False,\n                'active'    : True,\n                'moves'     : 1,\n                'vlan'      : 10,\n                'last_move' : 1454417742.58\n            },\n            {\n                'mac'       : '8c:60:4f:58:e1:c1',\n                'interface' : 'xe-1/0/1',\n                'static'    : False,\n                'active'    : True,\n                'moves'     : 2,\n                'vlan'      : 42,\n                'last_move' : 1453191948.11\n            }\n        ]\n    "
    proxy_output = salt.utils.napalm.call(napalm_device, 'get_mac_address_table', **{})
    if not proxy_output.get('result'):
        return proxy_output
    mac_address_table = proxy_output.get('out')
    if vlan and isinstance(vlan, int):
        mac_address_table = _filter_list(mac_address_table, 'vlan', vlan)
    if address:
        mac_address_table = _filter_list(mac_address_table, 'mac', address)
    if interface:
        mac_address_table = _filter_list(mac_address_table, 'interface', interface)
    proxy_output.update({'out': mac_address_table})
    return proxy_output

@salt.utils.napalm.proxy_napalm_wrap
def config(source=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2017.7.0\n\n    Return the whole configuration of the network device. By default, it will\n    return all possible configuration sources supported by the network device.\n    At most, there will be:\n\n    - running config\n    - startup config\n    - candidate config\n\n    To return only one of the configurations, you can use the ``source``\n    argument.\n\n    source\n        Which configuration type you want to display, default is all of them.\n\n        Options:\n\n        - running\n        - candidate\n        - startup\n\n    :return:\n        The object returned is a dictionary with the following keys:\n\n        - running (string): Representation of the native running configuration.\n        - candidate (string): Representation of the native candidate configuration.\n            If the device doesn't differentiate between running and startup\n            configuration this will an empty string.\n        - startup (string): Representation of the native startup configuration.\n            If the device doesn't differentiate between running and startup\n            configuration this will an empty string.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' net.config\n        salt '*' net.config source=candidate\n    "
    return salt.utils.napalm.call(napalm_device, 'get_config', **{'retrieve': source})

@salt.utils.napalm.proxy_napalm_wrap
def optics(**kwargs):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2017.7.0\n\n    Fetches the power usage on the various transceivers installed\n    on the network device (in dBm), and returns a view that conforms with the\n    OpenConfig model openconfig-platform-transceiver.yang.\n\n    :return:\n        Returns a dictionary where the keys are as listed below:\n            * intf_name (unicode)\n                * physical_channels\n                    * channels (list of dicts)\n                        * index (int)\n                        * state\n                            * input_power\n                                * instant (float)\n                                * avg (float)\n                                * min (float)\n                                * max (float)\n                            * output_power\n                                * instant (float)\n                                * avg (float)\n                                * min (float)\n                                * max (float)\n                            * laser_bias_current\n                                * instant (float)\n                                * avg (float)\n                                * min (float)\n                                * max (float)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' net.optics\n    "
    return salt.utils.napalm.call(napalm_device, 'get_optics', **{})

@salt.utils.napalm.proxy_napalm_wrap
def load_config(filename=None, text=None, test=False, commit=True, debug=False, replace=False, commit_in=None, commit_at=None, revert_in=None, revert_at=None, commit_jid=None, inherit_napalm_device=None, saltenv='base', **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Applies configuration changes on the device. It can be loaded from a file or from inline string.\n    If you send both a filename and a string containing the configuration, the file has higher precedence.\n\n    By default this function will commit the changes. If there are no changes, it does not commit and\n    the flag ``already_configured`` will be set as ``True`` to point this out.\n\n    To avoid committing the configuration, set the argument ``test`` to ``True`` and will discard (dry run).\n\n    To keep the changes but not commit, set ``commit`` to ``False``.\n\n    To replace the config, set ``replace`` to ``True``.\n\n    filename\n        Path to the file containing the desired configuration.\n        This can be specified using the absolute path to the file,\n        or using one of the following URL schemes:\n\n        - ``salt://``, to fetch the template from the Salt fileserver.\n        - ``http://`` or ``https://``\n        - ``ftp://``\n        - ``s3://``\n        - ``swift://``\n\n        .. versionchanged:: 2018.3.0\n\n    text\n        String containing the desired configuration.\n        This argument is ignored when ``filename`` is specified.\n\n    test: False\n        Dry run? If set as ``True``, will apply the config, discard and return the changes. Default: ``False``\n        and will commit the changes on the device.\n\n    commit: True\n        Commit? Default: ``True``.\n\n    debug: False\n        Debug mode. Will insert a new key under the output dictionary, as ``loaded_config`` containing the raw\n        configuration loaded on the device.\n\n        .. versionadded:: 2016.11.2\n\n    replace: False\n        Load and replace the configuration. Default: ``False``.\n\n        .. versionadded:: 2016.11.2\n\n    commit_in: ``None``\n        Commit the changes in a specific number of minutes / hours. Example of\n        accepted formats: ``5`` (commit in 5 minutes), ``2m`` (commit in 2\n        minutes), ``1h`` (commit the changes in 1 hour)`, ``5h30m`` (commit\n        the changes in 5 hours and 30 minutes).\n\n        .. note::\n            This feature works on any platforms, as it does not rely on the\n            native features of the network operating system.\n\n        .. note::\n            After the command is executed and the ``diff`` is not satisfactory,\n            or for any other reasons you have to discard the commit, you are\n            able to do so using the\n            :py:func:`net.cancel_commit <salt.modules.napalm_network.cancel_commit>`\n            execution function, using the commit ID returned by this function.\n\n        .. warning::\n            Using this feature, Salt will load the exact configuration you\n            expect, however the diff may change in time (i.e., if an user\n            applies a manual configuration change, or a different process or\n            command changes the configuration in the meanwhile).\n\n        .. versionadded:: 2019.2.0\n\n    commit_at: ``None``\n        Commit the changes at a specific time. Example of accepted formats:\n        ``1am`` (will commit the changes at the next 1AM), ``13:20`` (will\n        commit at 13:20), ``1:20am``, etc.\n\n        .. note::\n            This feature works on any platforms, as it does not rely on the\n            native features of the network operating system.\n\n        .. note::\n            After the command is executed and the ``diff`` is not satisfactory,\n            or for any other reasons you have to discard the commit, you are\n            able to do so using the\n            :py:func:`net.cancel_commit <salt.modules.napalm_network.cancel_commit>`\n            execution function, using the commit ID returned by this function.\n\n        .. warning::\n            Using this feature, Salt will load the exact configuration you\n            expect, however the diff may change in time (i.e., if an user\n            applies a manual configuration change, or a different process or\n            command changes the configuration in the meanwhile).\n\n        .. versionadded:: 2019.2.0\n\n    revert_in: ``None``\n        Commit and revert the changes in a specific number of minutes / hours.\n        Example of accepted formats: ``5`` (revert in 5 minutes), ``2m`` (revert\n        in 2 minutes), ``1h`` (revert the changes in 1 hour)`, ``5h30m`` (revert\n        the changes in 5 hours and 30 minutes).\n\n        .. note::\n            To confirm the commit, and prevent reverting the changes, you will\n            have to execute the\n            :mod:`net.confirm_commit <salt.modules.napalm_network.confirm_commit>`\n            function, using the commit ID returned by this function.\n\n        .. warning::\n            This works on any platform, regardless if they have or don\'t have\n            native capabilities to confirming a commit. However, please be\n            *very* cautious when using this feature: on Junos (as it is the only\n            NAPALM core platform supporting this natively) it executes a commit\n            confirmed as you would do from the command line.\n            All the other platforms don\'t have this capability natively,\n            therefore the revert is done via Salt. That means, your device needs\n            to be reachable at the moment when Salt will attempt to revert your\n            changes. Be cautious when pushing configuration changes that would\n            prevent you reach the device.\n\n            Similarly, if an user or a different process apply other\n            configuration changes in the meanwhile (between the moment you\n            commit and till the changes are reverted), these changes would be\n            equally reverted, as Salt cannot be aware of them.\n\n        .. versionadded:: 2019.2.0\n\n    revert_at: ``None``\n        Commit and revert the changes at a specific time. Example of accepted\n        formats: ``1am`` (will commit and revert the changes at the next 1AM),\n        ``13:20`` (will commit and revert at 13:20), ``1:20am``, etc.\n\n        .. note::\n            To confirm the commit, and prevent reverting the changes, you will\n            have to execute the\n            :mod:`net.confirm_commit <salt.modules.napalm_network.confirm_commit>`\n            function, using the commit ID returned by this function.\n\n        .. warning::\n            This works on any platform, regardless if they have or don\'t have\n            native capabilities to confirming a commit. However, please be\n            *very* cautious when using this feature: on Junos (as it is the only\n            NAPALM core platform supporting this natively) it executes a commit\n            confirmed as you would do from the command line.\n            All the other platforms don\'t have this capability natively,\n            therefore the revert is done via Salt. That means, your device needs\n            to be reachable at the moment when Salt will attempt to revert your\n            changes. Be cautious when pushing configuration changes that would\n            prevent you reach the device.\n\n            Similarly, if an user or a different process apply other\n            configuration changes in the meanwhile (between the moment you\n            commit and till the changes are reverted), these changes would be\n            equally reverted, as Salt cannot be aware of them.\n\n        .. versionadded:: 2019.2.0\n\n    saltenv: ``base``\n        Specifies the Salt environment name.\n\n        .. versionadded:: 2018.3.0\n\n    :return: a dictionary having the following keys:\n\n    * result (bool): if the config was applied successfully. It is ``False`` only in case of failure. In case     there are no changes to be applied and successfully performs all operations it is still ``True`` and so will be     the ``already_configured`` flag (example below)\n    * comment (str): a message for the user\n    * already_configured (bool): flag to check if there were no changes applied\n    * loaded_config (str): the configuration loaded on the device. Requires ``debug`` to be set as ``True``\n    * diff (str): returns the config changes applied\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' net.load_config text=\'ntp peer 192.168.0.1\'\n        salt \'*\' net.load_config filename=\'/absolute/path/to/your/file\'\n        salt \'*\' net.load_config filename=\'/absolute/path/to/your/file\' test=True\n        salt \'*\' net.load_config filename=\'/absolute/path/to/your/file\' commit=False\n\n    Example output:\n\n    .. code-block:: python\n\n        {\n            \'comment\': \'Configuration discarded.\',\n            \'already_configured\': False,\n            \'result\': True,\n            \'diff\': \'[edit interfaces xe-0/0/5]+   description "Adding a description";\'\n        }\n    '
    fun = 'load_merge_candidate'
    if replace:
        fun = 'load_replace_candidate'
    if salt.utils.napalm.not_always_alive(__opts__):
        napalm_device['CLOSE'] = False
    if filename:
        text = __salt__['cp.get_file_str'](filename, saltenv=saltenv)
        if text is False:
            ret = {'result': False, 'out': None}
            ret['comment'] = 'Unable to read from {}. Please specify a valid file or text.'.format(filename)
            log.error(ret['comment'])
            return ret
        if commit_jid:
            salt.utils.files.safe_rm(filename)
    _loaded = salt.utils.napalm.call(napalm_device, fun, **{'config': text})
    return _config_logic(napalm_device, _loaded, test=test, debug=debug, replace=replace, commit_config=commit, loaded_config=text, commit_at=commit_at, commit_in=commit_in, revert_in=revert_in, revert_at=revert_at, commit_jid=commit_jid, **kwargs)

@salt.utils.napalm.proxy_napalm_wrap
def load_template(template_name=None, template_source=None, context=None, defaults=None, template_engine='jinja', saltenv='base', template_hash=None, template_hash_name=None, skip_verify=False, test=False, commit=True, debug=False, replace=False, commit_in=None, commit_at=None, revert_in=None, revert_at=None, inherit_napalm_device=None, **template_vars):
    if False:
        for i in range(10):
            print('nop')
    '\n    Renders a configuration template (default: Jinja) and loads the result on the device.\n\n    By default this function will commit the changes. If there are no changes,\n    it does not commit, discards he config and the flag ``already_configured``\n    will be set as ``True`` to point this out.\n\n    To avoid committing the configuration, set the argument ``test`` to ``True``\n    and will discard (dry run).\n\n    To preserve the changes, set ``commit`` to ``False``.\n    However, this is recommended to be used only in exceptional cases\n    when there are applied few consecutive states\n    and/or configuration changes.\n    Otherwise the user might forget that the config DB is locked\n    and the candidate config buffer is not cleared/merged in the running config.\n\n    To replace the config, set ``replace`` to ``True``.\n\n    template_name\n        Identifies path to the template source.\n        The template can be either stored on the local machine, either remotely.\n        The recommended location is under the ``file_roots``\n        as specified in the master config file.\n        For example, let\'s suppose the ``file_roots`` is configured as:\n\n        .. code-block:: yaml\n\n            file_roots:\n              base:\n                - /etc/salt/states\n\n        Placing the template under ``/etc/salt/states/templates/example.jinja``,\n        it can be used as ``salt://templates/example.jinja``.\n        Alternatively, for local files, the user can specify the absolute path.\n        If remotely, the source can be retrieved via ``http``, ``https`` or ``ftp``.\n\n        Examples:\n\n        - ``salt://my_template.jinja``\n        - ``/absolute/path/to/my_template.jinja``\n        - ``http://example.com/template.cheetah``\n        - ``https:/example.com/template.mako``\n        - ``ftp://example.com/template.py``\n\n        .. versionchanged:: 2019.2.0\n            This argument can now support a list of templates to be rendered.\n            The resulting configuration text is loaded at once, as a single\n            configuration chunk.\n\n    template_source: None\n        Inline config template to be rendered and loaded on the device.\n\n    template_hash: None\n        Hash of the template file. Format: ``{hash_type: \'md5\', \'hsum\': <md5sum>}``\n\n        .. versionadded:: 2016.11.2\n\n    context: None\n        Overrides default context variables passed to the template.\n\n        .. versionadded:: 2019.2.0\n\n    template_hash_name: None\n        When ``template_hash`` refers to a remote file,\n        this specifies the filename to look for in that file.\n\n        .. versionadded:: 2016.11.2\n\n    saltenv: ``base``\n        Specifies the template environment.\n        This will influence the relative imports inside the templates.\n\n        .. versionadded:: 2016.11.2\n\n    template_engine: jinja\n        The following templates engines are supported:\n\n        - :mod:`cheetah<salt.renderers.cheetah>`\n        - :mod:`genshi<salt.renderers.genshi>`\n        - :mod:`jinja<salt.renderers.jinja>`\n        - :mod:`mako<salt.renderers.mako>`\n        - :mod:`py<salt.renderers.py>`\n        - :mod:`wempy<salt.renderers.wempy>`\n\n        .. versionadded:: 2016.11.2\n\n    skip_verify: True\n        If ``True``, hash verification of remote file sources\n        (``http://``, ``https://``, ``ftp://``) will be skipped,\n        and the ``source_hash`` argument will be ignored.\n\n        .. versionadded:: 2016.11.2\n\n    test: False\n        Dry run? If set to ``True``, will apply the config,\n        discard and return the changes.\n        Default: ``False`` and will commit the changes on the device.\n\n    commit: True\n        Commit? (default: ``True``)\n\n    debug: False\n        Debug mode. Will insert a new key under the output dictionary,\n        as ``loaded_config`` containing the raw result after the template was rendered.\n\n        .. versionadded:: 2016.11.2\n\n    replace: False\n        Load and replace the configuration.\n\n        .. versionadded:: 2016.11.2\n\n    commit_in: ``None``\n        Commit the changes in a specific number of minutes / hours. Example of\n        accepted formats: ``5`` (commit in 5 minutes), ``2m`` (commit in 2\n        minutes), ``1h`` (commit the changes in 1 hour)`, ``5h30m`` (commit\n        the changes in 5 hours and 30 minutes).\n\n        .. note::\n            This feature works on any platforms, as it does not rely on the\n            native features of the network operating system.\n\n        .. note::\n            After the command is executed and the ``diff`` is not satisfactory,\n            or for any other reasons you have to discard the commit, you are\n            able to do so using the\n            :py:func:`net.cancel_commit <salt.modules.napalm_network.cancel_commit>`\n            execution function, using the commit ID returned by this function.\n\n        .. warning::\n            Using this feature, Salt will load the exact configuration you\n            expect, however the diff may change in time (i.e., if an user\n            applies a manual configuration change, or a different process or\n            command changes the configuration in the meanwhile).\n\n        .. versionadded:: 2019.2.0\n\n    commit_at: ``None``\n        Commit the changes at a specific time. Example of accepted formats:\n        ``1am`` (will commit the changes at the next 1AM), ``13:20`` (will\n        commit at 13:20), ``1:20am``, etc.\n\n        .. note::\n            This feature works on any platforms, as it does not rely on the\n            native features of the network operating system.\n\n        .. note::\n            After the command is executed and the ``diff`` is not satisfactory,\n            or for any other reasons you have to discard the commit, you are\n            able to do so using the\n            :py:func:`net.cancel_commit <salt.modules.napalm_network.cancel_commit>`\n            execution function, using the commit ID returned by this function.\n\n        .. warning::\n            Using this feature, Salt will load the exact configuration you\n            expect, however the diff may change in time (i.e., if an user\n            applies a manual configuration change, or a different process or\n            command changes the configuration in the meanwhile).\n\n        .. versionadded:: 2019.2.0\n\n    revert_in: ``None``\n        Commit and revert the changes in a specific number of minutes / hours.\n        Example of accepted formats: ``5`` (revert in 5 minutes), ``2m`` (revert\n        in 2 minutes), ``1h`` (revert the changes in 1 hour)`, ``5h30m`` (revert\n        the changes in 5 hours and 30 minutes).\n\n        .. note::\n            To confirm the commit, and prevent reverting the changes, you will\n            have to execute the\n            :mod:`net.confirm_commit <salt.modules.napalm_network.confirm_commit>`\n            function, using the commit ID returned by this function.\n\n        .. warning::\n            This works on any platform, regardless if they have or don\'t have\n            native capabilities to confirming a commit. However, please be\n            *very* cautious when using this feature: on Junos (as it is the only\n            NAPALM core platform supporting this natively) it executes a commit\n            confirmed as you would do from the command line.\n            All the other platforms don\'t have this capability natively,\n            therefore the revert is done via Salt. That means, your device needs\n            to be reachable at the moment when Salt will attempt to revert your\n            changes. Be cautious when pushing configuration changes that would\n            prevent you reach the device.\n\n            Similarly, if an user or a different process apply other\n            configuration changes in the meanwhile (between the moment you\n            commit and till the changes are reverted), these changes would be\n            equally reverted, as Salt cannot be aware of them.\n\n        .. versionadded:: 2019.2.0\n\n    revert_at: ``None``\n        Commit and revert the changes at a specific time. Example of accepted\n        formats: ``1am`` (will commit and revert the changes at the next 1AM),\n        ``13:20`` (will commit and revert at 13:20), ``1:20am``, etc.\n\n        .. note::\n            To confirm the commit, and prevent reverting the changes, you will\n            have to execute the\n            :mod:`net.confirm_commit <salt.modules.napalm_network.confirm_commit>`\n            function, using the commit ID returned by this function.\n\n        .. warning::\n            This works on any platform, regardless if they have or don\'t have\n            native capabilities to confirming a commit. However, please be\n            *very* cautious when using this feature: on Junos (as it is the only\n            NAPALM core platform supporting this natively) it executes a commit\n            confirmed as you would do from the command line.\n            All the other platforms don\'t have this capability natively,\n            therefore the revert is done via Salt. That means, your device needs\n            to be reachable at the moment when Salt will attempt to revert your\n            changes. Be cautious when pushing configuration changes that would\n            prevent you reach the device.\n\n            Similarly, if an user or a different process apply other\n            configuration changes in the meanwhile (between the moment you\n            commit and till the changes are reverted), these changes would be\n            equally reverted, as Salt cannot be aware of them.\n\n        .. versionadded:: 2019.2.0\n\n    defaults: None\n        Default variables/context passed to the template.\n\n        .. versionadded:: 2016.11.2\n\n    template_vars\n        Dictionary with the arguments/context to be used when the template is rendered.\n\n        .. note::\n            Do not explicitly specify this argument. This represents any other\n            variable that will be sent to the template rendering system.\n            Please see the examples below!\n\n        .. note::\n            It is more recommended to use the ``context`` argument to avoid\n            conflicts between CLI arguments and template variables.\n\n    :return: a dictionary having the following keys:\n\n    - result (bool): if the config was applied successfully. It is ``False``\n      only in case of failure. In case there are no changes to be applied and\n      successfully performs all operations it is still ``True`` and so will be\n      the ``already_configured`` flag (example below)\n    - comment (str): a message for the user\n    - already_configured (bool): flag to check if there were no changes applied\n    - loaded_config (str): the configuration loaded on the device, after\n      rendering the template. Requires ``debug`` to be set as ``True``\n    - diff (str): returns the config changes applied\n\n    The template can use variables from the ``grains``, ``pillar`` or ``opts``, for example:\n\n    .. code-block:: jinja\n\n        {% set router_model = grains.get(\'model\') -%}\n        {% set router_vendor = grains.get(\'vendor\') -%}\n        {% set os_version = grains.get(\'version\') -%}\n        {% set hostname = pillar.get(\'proxy\', {}).get(\'host\') -%}\n        {% if router_vendor|lower == \'juniper\' %}\n        system {\n            host-name {{hostname}};\n        }\n        {% elif router_vendor|lower == \'cisco\' %}\n        hostname {{hostname}}\n        {% endif %}\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt \'*\' net.load_template set_ntp_peers peers=[192.168.0.1]  # uses NAPALM default templates\n\n        # inline template:\n        salt -G \'os:junos\' net.load_template template_source=\'system { host-name {{host_name}}; }\'         host_name=\'MX480.lab\'\n\n        # inline template using grains info:\n        salt -G \'os:junos\' net.load_template         template_source=\'system { host-name {{grains.model}}.lab; }\'\n        # if the device is a MX480, the command above will set the hostname as: MX480.lab\n\n        # inline template using pillar data:\n        salt -G \'os:junos\' net.load_template template_source=\'system { host-name {{pillar.proxy.host}}; }\'\n\n        salt \'*\' net.load_template https://bit.ly/2OhSgqP hostname=example  # will commit\n        salt \'*\' net.load_template https://bit.ly/2OhSgqP hostname=example test=True  # dry run\n\n        salt \'*\' net.load_template salt://templates/example.jinja debug=True  # Using the salt:// URI\n\n        # render a mako template:\n        salt \'*\' net.load_template salt://templates/example.mako template_engine=mako debug=True\n\n        # render remote template\n        salt -G \'os:junos\' net.load_template http://bit.ly/2fReJg7 test=True debug=True peers=[\'192.168.0.1\']\n        salt -G \'os:ios\' net.load_template http://bit.ly/2gKOj20 test=True debug=True peers=[\'192.168.0.1\']\n\n        # render multiple templates at once\n        salt \'*\' net.load_template "[\'https://bit.ly/2OhSgqP\', \'salt://templates/example.jinja\']" context="{\'hostname\': \'example\'}"\n\n    Example output:\n\n    .. code-block:: python\n\n        {\n            \'comment\': \'\',\n            \'already_configured\': False,\n            \'result\': True,\n            \'diff\': \'[edit system]+  host-name edge01.bjm01\',\n            \'loaded_config\': \'system { host-name edge01.bjm01; }\'\'\n        }\n    '
    _rendered = ''
    _loaded = {'result': True, 'comment': '', 'out': None}
    loaded_config = None
    if template_engine not in salt.utils.templates.TEMPLATE_REGISTRY:
        _loaded.update({'result': False, 'comment': 'Invalid templating engine! Choose between: {tpl_eng_opts}'.format(tpl_eng_opts=', '.join(list(salt.utils.templates.TEMPLATE_REGISTRY.keys())))})
        return _loaded
    salt_render_prefixes = ('salt://', 'http://', 'https://', 'ftp://')
    salt_render = False
    file_exists = False
    if not isinstance(template_name, (tuple, list)):
        for salt_render_prefix in salt_render_prefixes:
            if not salt_render:
                salt_render = salt_render or template_name.startswith(salt_render_prefix)
        file_exists = __salt__['file.file_exists'](template_name)
    if context is None:
        context = {}
    context.update(template_vars)
    if template_source:
        _rendered = __salt__['file.apply_template_on_contents'](contents=template_source, template=template_engine, context=context, defaults=defaults, saltenv=saltenv)
        if not isinstance(_rendered, str):
            if 'result' in _rendered:
                _loaded['result'] = _rendered['result']
            else:
                _loaded['result'] = False
            if 'comment' in _rendered:
                _loaded['comment'] = _rendered['comment']
            else:
                _loaded['comment'] = 'Error while rendering the template.'
            return _loaded
    else:
        if not isinstance(template_name, (list, tuple)):
            template_name = [template_name]
        if template_hash_name and (not isinstance(template_hash_name, (list, tuple))):
            template_hash_name = [template_hash_name]
        elif not template_hash_name:
            template_hash_name = [None] * len(template_name)
        if template_hash and isinstance(template_hash, str) and (not (template_hash.startswith('salt://') or template_hash.startswith('file://'))):
            template_hash = [template_hash]
        elif template_hash and isinstance(template_hash, str) and (template_hash.startswith('salt://') or template_hash.startswith('file://')):
            template_hash = [template_hash] * len(template_name)
        elif not template_hash:
            template_hash = [None] * len(template_name)
        for (tpl_index, tpl_name) in enumerate(template_name):
            tpl_hash = template_hash[tpl_index]
            tpl_hash_name = template_hash_name[tpl_index]
            _rand_filename = __salt__['random.hash'](tpl_name, 'md5')
            _temp_file = __salt__['file.join']('/tmp', _rand_filename)
            _managed = __salt__['file.get_managed'](name=_temp_file, source=tpl_name, source_hash=tpl_hash, source_hash_name=tpl_hash_name, user=None, group=None, mode=None, attrs=None, template=template_engine, context=context, defaults=defaults, saltenv=saltenv, skip_verify=skip_verify)
            if not isinstance(_managed, (list, tuple)) and isinstance(_managed, str):
                _loaded['comment'] += _managed
                _loaded['result'] = False
            elif isinstance(_managed, (list, tuple)) and (not len(_managed) > 0):
                _loaded['result'] = False
                _loaded['comment'] += 'Error while rendering the template.'
            elif isinstance(_managed, (list, tuple)) and (not len(_managed[0]) > 0):
                _loaded['result'] = False
                _loaded['comment'] += _managed[-1]
            if _loaded['result']:
                _temp_tpl_file = _managed[0]
                _temp_tpl_file_exists = __salt__['file.file_exists'](_temp_tpl_file)
                if not _temp_tpl_file_exists:
                    _loaded['result'] = False
                    _loaded['comment'] += 'Error while rendering the template.'
                    return _loaded
                _rendered += __salt__['file.read'](_temp_tpl_file)
                __salt__['file.remove'](_temp_tpl_file)
            else:
                return _loaded
    loaded_config = _rendered
    if _loaded['result']:
        fun = 'load_merge_candidate'
        if replace:
            fun = 'load_replace_candidate'
        if salt.utils.napalm.not_always_alive(__opts__):
            napalm_device['CLOSE'] = False
        _loaded = salt.utils.napalm.call(napalm_device, fun, **{'config': _rendered})
    return _config_logic(napalm_device, _loaded, test=test, debug=debug, replace=replace, commit_config=commit, loaded_config=loaded_config, commit_at=commit_at, commit_in=commit_in, revert_in=revert_in, revert_at=revert_at, **template_vars)

@salt.utils.napalm.proxy_napalm_wrap
def commit(inherit_napalm_device=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Commits the configuration changes made on the network device.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' net.commit\n    "
    return salt.utils.napalm.call(napalm_device, 'commit_config', **{})

@salt.utils.napalm.proxy_napalm_wrap
def discard_config(inherit_napalm_device=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Discards the changes applied.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' net.discard_config\n    "
    return salt.utils.napalm.call(napalm_device, 'discard_config', **{})

@salt.utils.napalm.proxy_napalm_wrap
def compare_config(inherit_napalm_device=None, **kwargs):
    if False:
        return 10
    "\n    Returns the difference between the running config and the candidate config.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' net.compare_config\n    "
    return salt.utils.napalm.call(napalm_device, 'compare_config', **{})

@salt.utils.napalm.proxy_napalm_wrap
def rollback(inherit_napalm_device=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Rollbacks the configuration.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' net.rollback\n    "
    return salt.utils.napalm.call(napalm_device, 'rollback', **{})

@salt.utils.napalm.proxy_napalm_wrap
def config_changed(inherit_napalm_device=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Will prompt if the configuration has been changed.\n\n    :return: A tuple with a boolean that specifies if the config was changed on the device.    And a string that provides more details of the reason why the configuration was not changed.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' net.config_changed\n    "
    is_config_changed = False
    reason = ''
    try_compare = compare_config(inherit_napalm_device=napalm_device)
    if try_compare.get('result'):
        if try_compare.get('out'):
            is_config_changed = True
        else:
            reason = 'Configuration was not changed on the device.'
    else:
        reason = try_compare.get('comment')
    return (is_config_changed, reason)

@salt.utils.napalm.proxy_napalm_wrap
def config_control(inherit_napalm_device=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Will check if the configuration was changed.\n    If differences found, will try to commit.\n    In case commit unsuccessful, will try to rollback.\n\n    :return: A tuple with a boolean that specifies if the config was changed/committed/rollbacked on the device.    And a string that provides more details of the reason why the configuration was not committed properly.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' net.config_control\n    "
    result = True
    comment = ''
    (changed, not_changed_rsn) = config_changed(inherit_napalm_device=napalm_device)
    if not changed:
        return (changed, not_changed_rsn)
    try_commit = commit()
    if not try_commit.get('result'):
        result = False
        comment = 'Unable to commit the changes: {reason}.\nWill try to rollback now!'.format(reason=try_commit.get('comment'))
        try_rollback = rollback()
        if not try_rollback.get('result'):
            comment += '\nCannot rollback! {reason}'.format(reason=try_rollback.get('comment'))
    return (result, comment)

def cancel_commit(jid):
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 2019.2.0\n\n    Cancel a commit scheduled to be executed via the ``commit_in`` and\n    ``commit_at`` arguments from the\n    :py:func:`net.load_template <salt.modules.napalm_network.load_template>` or\n    :py:func:`net.load_config <salt.modules.napalm_network.load_config>`\n    execution functions. The commit ID is displayed when the commit is scheduled\n    via the functions named above.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' net.cancel_commit 20180726083540640360\n    "
    job_name = '__napalm_commit_{}'.format(jid)
    removed = __salt__['schedule.delete'](job_name)
    if removed['result']:
        saved = __salt__['schedule.save']()
        removed['comment'] = 'Commit #{jid} cancelled.'.format(jid=jid)
    else:
        removed['comment'] = 'Unable to find commit #{jid}.'.format(jid=jid)
    return removed

def confirm_commit(jid):
    if False:
        return 10
    "\n    .. versionadded:: 2019.2.0\n\n    Confirm a commit scheduled to be reverted via the ``revert_in`` and\n    ``revert_at``  arguments from the\n    :mod:`net.load_template <salt.modules.napalm_network.load_template>` or\n    :mod:`net.load_config <salt.modules.napalm_network.load_config>`\n    execution functions. The commit ID is displayed when the commit confirmed\n    is scheduled via the functions named above.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' net.confirm_commit 20180726083540640360\n    "
    if __grains__['os'] == 'junos':
        confirmed = __salt__['napalm.junos_commit']()
        confirmed['result'] = confirmed.pop('out')
        confirmed['comment'] = confirmed.pop('message')
    else:
        confirmed = cancel_commit(jid)
    if confirmed['result']:
        confirmed['comment'] = 'Commit #{jid} confirmed.'.format(jid=jid)
    return confirmed

def save_config(source=None, path=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2019.2.0\n\n    Save the configuration to a file on the local file system.\n\n    source: ``running``\n        The configuration source. Choose from: ``running``, ``candidate``,\n        ``startup``. Default: ``running``.\n\n    path\n        Absolute path to file where to save the configuration.\n        To push the files to the Master, use\n        :mod:`cp.push <salt.modules.cp.push>` Execution function.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' net.save_config source=running\n    "
    if not source:
        source = 'running'
    if not path:
        path = salt.utils.files.mkstemp()
    running_config = __salt__['net.config'](source=source)
    if not running_config or not running_config['result']:
        log.error('Unable to retrieve the config')
        return running_config
    with salt.utils.files.fopen(path, 'w') as fh_:
        fh_.write(running_config['out'][source])
    return {'result': True, 'out': path, 'comment': '{source} config saved to {path}'.format(source=source, path=path)}

def replace_pattern(pattern, repl, count=0, flags=8, bufsize=1, append_if_not_found=False, prepend_if_not_found=False, not_found_content=None, search_only=False, show_changes=True, backslash_literal=False, source=None, path=None, test=False, replace=True, debug=False, commit=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    .. versionadded:: 2019.2.0\n\n    Replace occurrences of a pattern in the configuration source. If\n    ``show_changes`` is ``True``, then a diff of what changed will be returned,\n    otherwise a ``True`` will be returned when changes are made, and ``False``\n    when no changes are made.\n    This is a pure Python implementation that wraps Python\'s :py:func:`~re.sub`.\n\n    pattern\n        A regular expression, to be matched using Python\'s\n        :py:func:`~re.search`.\n\n    repl\n        The replacement text.\n\n    count: ``0``\n        Maximum number of pattern occurrences to be replaced. If count is a\n        positive integer ``n``, only ``n`` occurrences will be replaced,\n        otherwise all occurrences will be replaced.\n\n    flags (list or int): ``8``\n        A list of flags defined in the ``re`` module documentation from the\n        Python standard library. Each list item should be a string that will\n        correlate to the human-friendly flag name. E.g., ``[\'IGNORECASE\',\n        \'MULTILINE\']``. Optionally, ``flags`` may be an int, with a value\n        corresponding to the XOR (``|``) of all the desired flags. Defaults to\n        8 (which supports \'MULTILINE\').\n\n    bufsize (int or str): ``1``\n        How much of the configuration to buffer into memory at once. The\n        default value ``1`` processes one line at a time. The special value\n        ``file`` may be specified which will read the entire file into memory\n        before processing.\n\n    append_if_not_found: ``False``\n        If set to ``True``, and pattern is not found, then the content will be\n        appended to the file.\n\n    prepend_if_not_found: ``False``\n        If set to ``True`` and pattern is not found, then the content will be\n        prepended to the file.\n\n    not_found_content\n        Content to use for append/prepend if not found. If None (default), uses\n        ``repl``. Useful when ``repl`` uses references to group in pattern.\n\n    search_only: ``False``\n        If set to true, this no changes will be performed on the file, and this\n        function will simply return ``True`` if the pattern was matched, and\n        ``False`` if not.\n\n    show_changes: ``True``\n        If ``True``, return a diff of changes made. Otherwise, return ``True``\n        if changes were made, and ``False`` if not.\n\n    backslash_literal: ``False``\n        Interpret backslashes as literal backslashes for the repl and not\n        escape characters.  This will help when using append/prepend so that\n        the backslashes are not interpreted for the repl on the second run of\n        the state.\n\n    source: ``running``\n        The configuration source. Choose from: ``running``, ``candidate``, or\n        ``startup``. Default: ``running``.\n\n    path\n        Save the temporary configuration to a specific path, then read from\n        there.\n\n    test: ``False``\n        Dry run? If set as ``True``, will apply the config, discard and return\n        the changes. Default: ``False`` and will commit the changes on the\n        device.\n\n    commit: ``True``\n        Commit the configuration changes? Default: ``True``.\n\n    debug: ``False``\n        Debug mode. Will insert a new key in the output dictionary, as\n        ``loaded_config`` containing the raw configuration loaded on the device.\n\n    replace: ``True``\n        Load and replace the configuration. Default: ``True``.\n\n    If an equal sign (``=``) appears in an argument to a Salt command it is\n    interpreted as a keyword argument in the format ``key=val``. That\n    processing can be bypassed in order to pass an equal sign through to the\n    remote shell command by manually specifying the kwarg:\n\n    .. code-block:: bash\n\n        salt \'*\' net.replace_pattern "bind-address\\s*=" "bind-address:"\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' net.replace_pattern PREFIX-LIST_NAME new-prefix-list-name\n        salt \'*\' net.replace_pattern bgp-group-name new-bgp-group-name count=1\n    '
    config_saved = save_config(source=source, path=path)
    if not config_saved or not config_saved['result']:
        return config_saved
    path = config_saved['out']
    replace_pattern = __salt__['file.replace'](path, pattern, repl, count=count, flags=flags, bufsize=bufsize, append_if_not_found=append_if_not_found, prepend_if_not_found=prepend_if_not_found, not_found_content=not_found_content, search_only=search_only, show_changes=show_changes, backslash_literal=backslash_literal)
    with salt.utils.files.fopen(path, 'r') as fh_:
        updated_config = fh_.read()
    return __salt__['net.load_config'](text=updated_config, test=test, debug=debug, replace=replace, commit=commit)

def blockreplace(marker_start, marker_end, content='', append_if_not_found=False, prepend_if_not_found=False, show_changes=True, append_newline=False, source='running', path=None, test=False, commit=True, debug=False, replace=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2019.2.0\n\n    Replace content of the configuration source, delimited by the line markers.\n\n    A block of content delimited by comments can help you manage several lines\n    without worrying about old entries removal.\n\n    marker_start\n        The line content identifying a line as the start of the content block.\n        Note that the whole line containing this marker will be considered,\n        so whitespace or extra content before or after the marker is included\n        in final output.\n\n    marker_end\n        The line content identifying a line as the end of the content block.\n        Note that the whole line containing this marker will be considered,\n        so whitespace or extra content before or after the marker is included\n        in final output.\n\n    content\n        The content to be used between the two lines identified by\n        ``marker_start`` and ``marker_stop``.\n\n    append_if_not_found: ``False``\n        If markers are not found and set to True then, the markers and content\n        will be appended to the file.\n\n    prepend_if_not_found: ``False``\n        If markers are not found and set to True then, the markers and content\n        will be prepended to the file.\n\n    append_newline: ``False``\n        Controls whether or not a newline is appended to the content block.\n        If the value of this argument is ``True`` then a newline will be added\n        to the content block. If it is ``False``, then a newline will not be\n        added to the content block. If it is ``None`` then a newline will only\n        be added to the content block if it does not already end in a newline.\n\n    show_changes: ``True``\n        Controls how changes are presented. If ``True``, this function will\n        return the of the changes made.\n        If ``False``, then it will return a boolean (``True`` if any changes\n        were made, otherwise False).\n\n    source: ``running``\n        The configuration source. Choose from: ``running``, ``candidate``, or\n        ``startup``. Default: ``running``.\n\n    path: ``None``\n        Save the temporary configuration to a specific path, then read from\n        there. This argument is optional, can be used when you prefers a\n        particular location of the temporary file.\n\n    test: ``False``\n        Dry run? If set as ``True``, will apply the config, discard and return\n        the changes. Default: ``False`` and will commit the changes on the\n        device.\n\n    commit: ``True``\n        Commit the configuration changes? Default: ``True``.\n\n    debug: ``False``\n        Debug mode. Will insert a new key in the output dictionary, as\n        ``loaded_config`` containing the raw configuration loaded on the device.\n\n    replace: ``True``\n        Load and replace the configuration. Default: ``True``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' net.blockreplace 'ntp' 'interface' ''\n    "
    config_saved = save_config(source=source, path=path)
    if not config_saved or not config_saved['result']:
        return config_saved
    path = config_saved['out']
    replace_pattern = __salt__['file.blockreplace'](path, marker_start=marker_start, marker_end=marker_end, content=content, append_if_not_found=append_if_not_found, prepend_if_not_found=prepend_if_not_found, show_changes=show_changes, append_newline=append_newline)
    with salt.utils.files.fopen(path, 'r') as fh_:
        updated_config = fh_.read()
    return __salt__['net.load_config'](text=updated_config, test=test, debug=debug, replace=replace, commit=commit)

def patch(patchfile, options='', saltenv='base', source_hash=None, show_changes=True, source='running', path=None, test=False, commit=True, debug=False, replace=True):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2019.2.0\n\n    Apply a patch to the configuration source, and load the result into the\n    running config of the device.\n\n    patchfile\n        A patch file to apply to the configuration source.\n\n    options\n        Options to pass to patch.\n\n    source_hash\n        If the patch file (specified via the ``patchfile`` argument)  is an\n        HTTP(S) or FTP URL and the file exists in the minion's file cache, this\n        option can be passed to keep the minion from re-downloading the file if\n        the cached copy matches the specified hash.\n\n    show_changes: ``True``\n        Controls how changes are presented. If ``True``, this function will\n        return the of the changes made.\n        If ``False``, then it will return a boolean (``True`` if any changes\n        were made, otherwise False).\n\n    source: ``running``\n        The configuration source. Choose from: ``running``, ``candidate``, or\n        ``startup``. Default: ``running``.\n\n    path: ``None``\n        Save the temporary configuration to a specific path, then read from\n        there. This argument is optional, can the user prefers a particular\n        location of the temporary file.\n\n    test: ``False``\n        Dry run? If set as ``True``, will apply the config, discard and return\n        the changes. Default: ``False`` and will commit the changes on the\n        device.\n\n    commit: ``True``\n        Commit the configuration changes? Default: ``True``.\n\n    debug: ``False``\n        Debug mode. Will insert a new key in the output dictionary, as\n        ``loaded_config`` containing the raw configuration loaded on the device.\n\n    replace: ``True``\n        Load and replace the configuration. Default: ``True``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' net.patch https://example.com/running_config.patch\n    "
    config_saved = save_config(source=source, path=path)
    if not config_saved or not config_saved['result']:
        return config_saved
    path = config_saved['out']
    patchfile_cache = __salt__['cp.cache_file'](patchfile)
    if patchfile_cache is False:
        return {'out': None, 'result': False, 'comment': 'The file "{}" does not exist.'.format(patchfile)}
    replace_pattern = __salt__['file.patch'](path, patchfile_cache, options=options)
    with salt.utils.files.fopen(path, 'r') as fh_:
        updated_config = fh_.read()
    return __salt__['net.load_config'](text=updated_config, test=test, debug=debug, replace=replace, commit=commit)