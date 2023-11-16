"""
TextFSM
=======

.. versionadded:: 2018.3.0

Execution module that processes plain text and extracts data
using TextFSM templates. The output is presented in JSON serializable
data, and can be easily re-used in other modules, or directly
inside the renderer (Jinja, Mako, Genshi, etc.).

:depends:   - textfsm Python library

.. note::

    Install  ``textfsm`` library: ``pip install textfsm``.
"""
import logging
import os
from salt.utils.files import fopen
try:
    import textfsm
    HAS_TEXTFSM = True
except ImportError:
    HAS_TEXTFSM = False
try:
    from textfsm import clitable
    HAS_CLITABLE = True
except ImportError:
    HAS_CLITABLE = False
log = logging.getLogger(__name__)
__virtualname__ = 'textfsm'
__proxyenabled__ = ['*']

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only load this execution module if TextFSM is installed.\n    '
    if HAS_TEXTFSM:
        return __virtualname__
    return (False, 'The textfsm execution module failed to load: requires the textfsm library.')

def _clitable_to_dict(objects, fsm_handler):
    if False:
        i = 10
        return i + 15
    '\n    Converts TextFSM cli_table object to list of dictionaries.\n    '
    objs = []
    log.debug('Cli Table: %s; FSM handler: %s', objects, fsm_handler)
    for row in objects:
        temp_dict = {}
        for (index, element) in enumerate(row):
            temp_dict[fsm_handler.header[index].lower()] = element
        objs.append(temp_dict)
    log.debug('Extraction result: %s', objs)
    return objs

def extract(template_path, raw_text=None, raw_text_file=None, saltenv='base'):
    if False:
        i = 10
        return i + 15
    '\n    Extracts the data entities from the unstructured\n    raw text sent as input and returns the data\n    mapping, processing using the TextFSM template.\n\n    template_path\n        The path to the TextFSM template.\n        This can be specified using the absolute path\n        to the file, or using one of the following URL schemes:\n\n        - ``salt://``, to fetch the template from the Salt fileserver.\n        - ``http://`` or ``https://``\n        - ``ftp://``\n        - ``s3://``\n        - ``swift://``\n\n    raw_text: ``None``\n        The unstructured text to be parsed.\n\n    raw_text_file: ``None``\n        Text file to read, having the raw text to be parsed using the TextFSM template.\n        Supports the same URL schemes as the ``template_path`` argument.\n\n    saltenv: ``base``\n        Salt fileserver environment from which to retrieve the file.\n        Ignored if ``template_path`` is not a ``salt://`` URL.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' textfsm.extract salt://textfsm/juniper_version_template raw_text_file=s3://junos_ver.txt\n        salt \'*\' textfsm.extract http://some-server/textfsm/juniper_version_template raw_text=\'Hostname: router.abc ... snip ...\'\n\n    Jinja template example:\n\n    .. code-block:: jinja\n\n        {%- set raw_text = \'Hostname: router.abc ... snip ...\' -%}\n        {%- set textfsm_extract = salt.textfsm.extract(\'https://some-server/textfsm/juniper_version_template\', raw_text) -%}\n\n    Raw text example:\n\n    .. code-block:: text\n\n        Hostname: router.abc\n        Model: mx960\n        JUNOS Base OS boot [9.1S3.5]\n        JUNOS Base OS Software Suite [9.1S3.5]\n        JUNOS Kernel Software Suite [9.1S3.5]\n        JUNOS Crypto Software Suite [9.1S3.5]\n        JUNOS Packet Forwarding Engine Support (M/T Common) [9.1S3.5]\n        JUNOS Packet Forwarding Engine Support (MX Common) [9.1S3.5]\n        JUNOS Online Documentation [9.1S3.5]\n        JUNOS Routing Software Suite [9.1S3.5]\n\n    TextFSM Example:\n\n    .. code-block:: text\n\n        Value Chassis (\\S+)\n        Value Required Model (\\S+)\n        Value Boot (.*)\n        Value Base (.*)\n        Value Kernel (.*)\n        Value Crypto (.*)\n        Value Documentation (.*)\n        Value Routing (.*)\n\n        Start\n        # Support multiple chassis systems.\n          ^\\S+:$$ -> Continue.Record\n          ^${Chassis}:$$\n          ^Model: ${Model}\n          ^JUNOS Base OS boot \\[${Boot}\\]\n          ^JUNOS Software Release \\[${Base}\\]\n          ^JUNOS Base OS Software Suite \\[${Base}\\]\n          ^JUNOS Kernel Software Suite \\[${Kernel}\\]\n          ^JUNOS Crypto Software Suite \\[${Crypto}\\]\n          ^JUNOS Online Documentation \\[${Documentation}\\]\n          ^JUNOS Routing Software Suite \\[${Routing}\\]\n\n    Output example:\n\n    .. code-block:: json\n\n        {\n            "comment": "",\n            "result": true,\n            "out": [\n                {\n                    "kernel": "9.1S3.5",\n                    "documentation": "9.1S3.5",\n                    "boot": "9.1S3.5",\n                    "crypto": "9.1S3.5",\n                    "chassis": "",\n                    "routing": "9.1S3.5",\n                    "base": "9.1S3.5",\n                    "model": "mx960"\n                }\n            ]\n        }\n    '
    ret = {'result': False, 'comment': '', 'out': None}
    log.debug('Caching %s(saltenv: %s) using the Salt fileserver', template_path, saltenv)
    tpl_cached_path = __salt__['cp.cache_file'](template_path, saltenv=saltenv)
    if tpl_cached_path is False:
        ret['comment'] = 'Unable to read the TextFSM template from {}'.format(template_path)
        log.error(ret['comment'])
        return ret
    try:
        log.debug('Reading TextFSM template from cache path: %s', tpl_cached_path)
        tpl_file_handle = fopen(tpl_cached_path, 'r')
        log.debug(tpl_file_handle.read())
        tpl_file_handle.seek(0)
        fsm_handler = textfsm.TextFSM(tpl_file_handle)
    except textfsm.TextFSMTemplateError as tfte:
        log.error('Unable to parse the TextFSM template', exc_info=True)
        ret['comment'] = 'Unable to parse the TextFSM template from {}: {}. Please check the logs.'.format(template_path, tfte)
        return ret
    if not raw_text and raw_text_file:
        log.debug('Trying to read the raw input from %s', raw_text_file)
        raw_text = __salt__['cp.get_file_str'](raw_text_file, saltenv=saltenv)
        if raw_text is False:
            ret['comment'] = 'Unable to read from {}. Please specify a valid input file or text.'.format(raw_text_file)
            log.error(ret['comment'])
            return ret
    if not raw_text:
        ret['comment'] = 'Please specify a valid input file or text.'
        log.error(ret['comment'])
        return ret
    log.debug('Processing the raw text:\n%s', raw_text)
    objects = fsm_handler.ParseText(raw_text)
    ret['out'] = _clitable_to_dict(objects, fsm_handler)
    ret['result'] = True
    return ret

def index(command, platform=None, platform_grain_name=None, platform_column_name=None, output=None, output_file=None, textfsm_path=None, index_file=None, saltenv='base', include_empty=False, include_pat=None, exclude_pat=None):
    if False:
        return 10
    "\n    Dynamically identify the template required to extract the\n    information from the unstructured raw text.\n\n    The output has the same structure as the ``extract`` execution\n    function, the difference being that ``index`` is capable\n    to identify what template to use, based on the platform\n    details and the ``command``.\n\n    command\n        The command executed on the device, to get the output.\n\n    platform\n        The platform name, as defined in the TextFSM index file.\n\n        .. note::\n            For ease of use, it is recommended to define the TextFSM\n            indexfile with values that can be matches using the grains.\n\n    platform_grain_name\n        The name of the grain used to identify the platform name\n        in the TextFSM index file.\n\n        .. note::\n            This option can be also specified in the minion configuration\n            file or pillar as ``textfsm_platform_grain``.\n\n        .. note::\n            This option is ignored when ``platform`` is specified.\n\n    platform_column_name: ``Platform``\n        The column name used to identify the platform,\n        exactly as specified in the TextFSM index file.\n        Default: ``Platform``.\n\n        .. note::\n            This is field is case sensitive, make sure\n            to assign the correct value to this option,\n            exactly as defined in the index file.\n\n        .. note::\n            This option can be also specified in the minion configuration\n            file or pillar as ``textfsm_platform_column_name``.\n\n    output\n        The raw output from the device, to be parsed\n        and extract the structured data.\n\n    output_file\n        The path to a file that contains the raw output from the device,\n        used to extract the structured data.\n        This option supports the usual Salt-specific schemes: ``file://``,\n        ``salt://``, ``http://``, ``https://``, ``ftp://``, ``s3://``, ``swift://``.\n\n    textfsm_path\n        The path where the TextFSM templates can be found. This can be either\n        absolute path on the server, either specified using the following URL\n        schemes: ``file://``, ``salt://``, ``http://``, ``https://``, ``ftp://``,\n        ``s3://``, ``swift://``.\n\n        .. note::\n            This needs to be a directory with a flat structure, having an\n            index file (whose name can be specified using the ``index_file`` option)\n            and a number of TextFSM templates.\n\n        .. note::\n            This option can be also specified in the minion configuration\n            file or pillar as ``textfsm_path``.\n\n    index_file: ``index``\n        The name of the TextFSM index file, under the ``textfsm_path``. Default: ``index``.\n\n        .. note::\n            This option can be also specified in the minion configuration\n            file or pillar as ``textfsm_index_file``.\n\n    saltenv: ``base``\n        Salt fileserver environment from which to retrieve the file.\n        Ignored if ``textfsm_path`` is not a ``salt://`` URL.\n\n    include_empty: ``False``\n        Include empty files under the ``textfsm_path``.\n\n    include_pat\n        Glob or regex to narrow down the files cached from the given path.\n        If matching with a regex, the regex must be prefixed with ``E@``,\n        otherwise the expression will be interpreted as a glob.\n\n    exclude_pat\n        Glob or regex to exclude certain files from being cached from the given path.\n        If matching with a regex, the regex must be prefixed with ``E@``,\n        otherwise the expression will be interpreted as a glob.\n\n        .. note::\n            If used with ``include_pat``, files matching this pattern will be\n            excluded from the subset of files defined by ``include_pat``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' textfsm.index 'sh ver' platform=Juniper output_file=salt://textfsm/juniper_version_example textfsm_path=salt://textfsm/\n        salt '*' textfsm.index 'sh ver' output_file=salt://textfsm/juniper_version_example textfsm_path=ftp://textfsm/ platform_column_name=Vendor\n        salt '*' textfsm.index 'sh ver' output_file=salt://textfsm/juniper_version_example textfsm_path=https://some-server/textfsm/ platform_column_name=Vendor platform_grain_name=vendor\n\n    TextFSM index file example:\n\n    ``salt://textfsm/index``\n\n    .. code-block:: text\n\n        Template, Hostname, Vendor, Command\n        juniper_version_template, .*, Juniper, sh[[ow]] ve[[rsion]]\n\n    The usage can be simplified,\n    by defining (some of) the following options: ``textfsm_platform_grain``,\n    ``textfsm_path``, ``textfsm_platform_column_name``, or ``textfsm_index_file``,\n    in the (proxy) minion configuration file or pillar.\n\n    Configuration example:\n\n    .. code-block:: yaml\n\n        textfsm_platform_grain: vendor\n        textfsm_path: salt://textfsm/\n        textfsm_platform_column_name: Vendor\n\n    And the CLI usage becomes as simple as:\n\n    .. code-block:: bash\n\n        salt '*' textfsm.index 'sh ver' output_file=salt://textfsm/juniper_version_example\n\n    Usgae inside a Jinja template:\n\n    .. code-block:: jinja\n\n        {%- set command = 'sh ver' -%}\n        {%- set output = salt.net.cli(command) -%}\n        {%- set textfsm_extract = salt.textfsm.index(command, output=output) -%}\n    "
    ret = {'out': None, 'result': False, 'comment': ''}
    if not HAS_CLITABLE:
        ret['comment'] = 'TextFSM does not seem that has clitable embedded.'
        log.error(ret['comment'])
        return ret
    if not platform:
        platform_grain_name = __opts__.get('textfsm_platform_grain') or __pillar__.get('textfsm_platform_grain', platform_grain_name)
        if platform_grain_name:
            log.debug('Using the %s grain to identify the platform name', platform_grain_name)
            platform = __grains__.get(platform_grain_name)
            if not platform:
                ret['comment'] = 'Unable to identify the platform name using the {} grain.'.format(platform_grain_name)
                return ret
            log.info('Using platform: %s', platform)
        else:
            ret['comment'] = 'No platform specified, no platform grain identifier configured.'
            log.error(ret['comment'])
            return ret
    if not textfsm_path:
        log.debug('No TextFSM templates path specified, trying to look into the opts and pillar')
        textfsm_path = __opts__.get('textfsm_path') or __pillar__.get('textfsm_path')
        if not textfsm_path:
            ret['comment'] = 'No TextFSM templates path specified. Please configure in opts/pillar/function args.'
            log.error(ret['comment'])
            return ret
    log.debug('Caching %s(saltenv: %s) using the Salt fileserver', textfsm_path, saltenv)
    textfsm_cachedir_ret = __salt__['cp.cache_dir'](textfsm_path, saltenv=saltenv, include_empty=include_empty, include_pat=include_pat, exclude_pat=exclude_pat)
    log.debug('Cache fun return:\n%s', textfsm_cachedir_ret)
    if not textfsm_cachedir_ret:
        ret['comment'] = 'Unable to fetch from {}. Is the TextFSM path correctly specified?'.format(textfsm_path)
        log.error(ret['comment'])
        return ret
    textfsm_cachedir = os.path.dirname(textfsm_cachedir_ret[0])
    index_file = __opts__.get('textfsm_index_file') or __pillar__.get('textfsm_index_file', 'index')
    index_file_path = os.path.join(textfsm_cachedir, index_file)
    log.debug('Using the cached index file: %s', index_file_path)
    log.debug('TextFSM templates cached under: %s', textfsm_cachedir)
    textfsm_obj = clitable.CliTable(index_file_path, textfsm_cachedir)
    attrs = {'Command': command}
    platform_column_name = __opts__.get('textfsm_platform_column_name') or __pillar__.get('textfsm_platform_column_name', 'Platform')
    log.info('Using the TextFSM platform idenfiticator: %s', platform_column_name)
    attrs[platform_column_name] = platform
    log.debug('Processing the TextFSM index file using the attributes: %s', attrs)
    if not output and output_file:
        log.debug('Processing the output from %s', output_file)
        output = __salt__['cp.get_file_str'](output_file, saltenv=saltenv)
        if output is False:
            ret['comment'] = 'Unable to read from {}. Please specify a valid file or text.'.format(output_file)
            log.error(ret['comment'])
            return ret
    if not output:
        ret['comment'] = 'Please specify a valid output text or file'
        log.error(ret['comment'])
        return ret
    log.debug('Processing the raw text:\n%s', output)
    try:
        textfsm_obj.ParseCmd(output, attrs)
        ret['out'] = _clitable_to_dict(textfsm_obj, textfsm_obj)
        ret['result'] = True
    except clitable.CliTableError as cterr:
        log.error('Unable to proces the CliTable', exc_info=True)
        ret['comment'] = 'Unable to process the output: {}'.format(cterr)
    return ret