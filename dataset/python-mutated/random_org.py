"""
Module for retrieving random information from Random.org

.. versionadded:: 2015.5.0

:configuration: This module can be used by either passing an api key and version
    directly or by specifying both in a configuration profile in the salt
    master/minion config.

    For example:

    .. code-block:: yaml

        random_org:
          api_key: 7be1402d-5719-5bd3-a306-3def9f135da5
          api_version: 1
"""
import http.client
import logging
import urllib.request
import salt.utils.http
import salt.utils.json
log = logging.getLogger(__name__)
__virtualname__ = 'random_org'
RANDOM_ORG_FUNCTIONS = {'1': {'getUsage': {'method': 'getUsage'}, 'generateIntegers': {'method': 'generateIntegers'}, 'generateStrings': {'method': 'generateStrings'}, 'generateUUIDs': {'method': 'generateUUIDs'}, 'generateDecimalFractions': {'method': 'generateDecimalFractions'}, 'generateGaussians': {'method': 'generateGaussians'}, 'generateBlobs': {'method': 'generateBlobs'}}}

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Return virtual name of the module.\n\n    :return: The virtual name of the module.\n    '
    return __virtualname__

def _numeric(n):
    if False:
        i = 10
        return i + 15
    '\n    Tell whether an argument is numeric\n    '
    return isinstance(n, (int, float))

def _query(api_version=None, data=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Slack object method function to construct and execute on the API URL.\n\n    :param api_key:     The Random.org api key.\n    :param api_version: The version of Random.org api.\n    :param data:        The data to be sent for POST method.\n    :return:            The json response from the API call or False.\n    '
    if data is None:
        data = {}
    ret = {'res': True}
    api_url = 'https://api.random.org/'
    base_url = urllib.parse.urljoin(api_url, 'json-rpc/' + str(api_version) + '/invoke')
    data = salt.utils.json.dumps(data)
    result = salt.utils.http.query(base_url, method='POST', params={}, data=data, decode=True, status=True, header_dict={}, opts=__opts__)
    if result.get('status', None) == http.client.OK:
        _result = result['dict']
        if _result.get('result'):
            return _result.get('result')
        if _result.get('error'):
            return _result.get('error')
        return False
    elif result.get('status', None) == http.client.NO_CONTENT:
        return False
    else:
        ret['message'] = result.text if hasattr(result, 'text') else ''
        return ret

def getUsage(api_key=None, api_version=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Show current usages statistics\n\n    :param api_key: The Random.org api key.\n    :param api_version: The Random.org api version.\n    :return: The current usage statistics.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' random_org.getUsage\n\n        salt '*' random_org.getUsage api_key=peWcBiMOS9HrZG15peWcBiMOS9HrZG15 api_version=1\n    "
    ret = {'res': True}
    if not api_key or not api_version:
        try:
            options = __salt__['config.option']('random_org')
            if not api_key:
                api_key = options.get('api_key')
            if not api_version:
                api_version = options.get('api_version')
        except (NameError, KeyError, AttributeError):
            log.error('No Random.org api key found.')
            ret['message'] = 'No Random.org api key or api version found.'
            ret['res'] = False
            return ret
    if isinstance(api_version, int):
        api_version = str(api_version)
    _function = RANDOM_ORG_FUNCTIONS.get(api_version).get('getUsage').get('method')
    data = {}
    data['id'] = 1911220
    data['jsonrpc'] = '2.0'
    data['method'] = _function
    data['params'] = {'apiKey': api_key}
    result = _query(api_version=api_version, data=data)
    if result:
        ret['bitsLeft'] = result.get('bitsLeft')
        ret['requestsLeft'] = result.get('requestsLeft')
        ret['totalBits'] = result.get('totalBits')
        ret['totalRequests'] = result.get('totalRequests')
    else:
        ret['res'] = False
        ret['message'] = result['message']
    return ret

def generateIntegers(api_key=None, api_version=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Generate random integers\n\n    :param api_key: The Random.org api key.\n    :param api_version: The Random.org api version.\n    :param number: The number of integers to generate\n    :param minimum: The lower boundary for the range from which the\n                    random numbers will be picked. Must be within\n                    the [-1e9,1e9] range.\n    :param maximum: The upper boundary for the range from which the\n                    random numbers will be picked. Must be within\n                    the [-1e9,1e9] range.\n    :param replacement: Specifies whether the random numbers should\n                        be picked with replacement. The default (true)\n                        will cause the numbers to be picked with replacement,\n                        i.e., the resulting numbers may contain duplicate\n                        values (like a series of dice rolls). If you want the\n                        numbers picked to be unique (like raffle tickets drawn\n                        from a container), set this value to false.\n    :param base: Specifies the base that will be used to display the numbers.\n                 Values allowed are 2, 8, 10 and 16. This affects the JSON\n                 types and formatting of the resulting data as discussed below.\n    :return: A list of integers.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' random_org.generateIntegers number=5 minimum=1 maximum=6\n\n        salt '*' random_org.generateIntegers number=5 minimum=2 maximum=255 base=2\n\n    "
    ret = {'res': True}
    if not api_key or not api_version:
        try:
            options = __salt__['config.option']('random_org')
            if not api_key:
                api_key = options.get('api_key')
            if not api_version:
                api_version = options.get('api_version')
        except (NameError, KeyError, AttributeError):
            log.error('No Random.org api key found.')
            ret['message'] = 'No Random.org api key or api version found.'
            ret['res'] = False
            return ret
    for item in ['number', 'minimum', 'maximum']:
        if item not in kwargs:
            ret['res'] = False
            ret['message'] = 'Rquired argument, {} is missing.'.format(item)
            return ret
    if not _numeric(kwargs['number']) or not 1 <= kwargs['number'] <= 10000:
        ret['res'] = False
        ret['message'] = 'Number of integers must be between 1 and 10000'
        return ret
    if not _numeric(kwargs['minimum']) or not -1000000000 <= kwargs['minimum'] <= 1000000000:
        ret['res'] = False
        ret['message'] = 'Minimum argument must be between -1,000,000,000 and 1,000,000,000'
        return ret
    if not _numeric(kwargs['maximum']) or not -1000000000 <= kwargs['maximum'] <= 1000000000:
        ret['res'] = False
        ret['message'] = 'Maximum argument must be between -1,000,000,000 and 1,000,000,000'
        return ret
    if 'base' in kwargs:
        base = kwargs['base']
        if base not in [2, 8, 10, 16]:
            ret['res'] = False
            ret['message'] = 'Base must be either 2, 8, 10 or 16.'
            return ret
    else:
        base = 10
    if 'replacement' not in kwargs:
        replacement = True
    else:
        replacement = kwargs['replacement']
    if isinstance(api_version, int):
        api_version = str(api_version)
    _function = RANDOM_ORG_FUNCTIONS.get(api_version).get('generateIntegers').get('method')
    data = {}
    data['id'] = 1911220
    data['jsonrpc'] = '2.0'
    data['method'] = _function
    data['params'] = {'apiKey': api_key, 'n': kwargs['number'], 'min': kwargs['minimum'], 'max': kwargs['maximum'], 'replacement': replacement, 'base': base}
    result = _query(api_version=api_version, data=data)
    log.debug('result %s', result)
    if result:
        if 'random' in result:
            random_data = result.get('random').get('data')
            ret['data'] = random_data
        else:
            ret['res'] = False
            ret['message'] = result['message']
    else:
        ret['res'] = False
        ret['message'] = result['message']
    return ret

def generateStrings(api_key=None, api_version=None, **kwargs):
    if False:
        return 10
    "\n    Generate random strings.\n\n    :param api_key: The Random.org api key.\n    :param api_version: The Random.org api version.\n    :param number: The number of strings to generate.\n    :param length: The length of each string. Must be\n                   within the [1,20] range. All strings\n                   will be of the same length\n    :param characters: A string that contains the set of\n                       characters that are allowed to occur\n                       in the random strings. The maximum number\n                       of characters is 80.\n    :param replacement: Specifies whether the random strings should be picked\n                        with replacement. The default (true) will cause the\n                        strings to be picked with replacement, i.e., the\n                        resulting list of strings may contain duplicates (like\n                        a series of dice rolls). If you want the strings to be\n                        unique (like raffle tickets drawn from a container), set\n                        this value to false.\n    :return: A list of strings.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' random_org.generateStrings number=5 length=8 characters='abcdefghijklmnopqrstuvwxyz'\n\n        salt '*' random_org.generateStrings number=10 length=16 characters'abcdefghijklmnopqrstuvwxyz'\n\n    "
    ret = {'res': True}
    if not api_key or not api_version:
        try:
            options = __salt__['config.option']('random_org')
            if not api_key:
                api_key = options.get('api_key')
            if not api_version:
                api_version = options.get('api_version')
        except (NameError, KeyError, AttributeError):
            log.error('No Random.org api key found.')
            ret['message'] = 'No Random.org api key or api version found.'
            ret['res'] = False
            return ret
    for item in ['number', 'length', 'characters']:
        if item not in kwargs:
            ret['res'] = False
            ret['message'] = 'Required argument, {} is missing.'.format(item)
            return ret
    if not _numeric(kwargs['number']) or not 1 <= kwargs['number'] <= 10000:
        ret['res'] = False
        ret['message'] = 'Number of strings must be between 1 and 10000'
        return ret
    if not _numeric(kwargs['length']) or not 1 <= kwargs['length'] <= 20:
        ret['res'] = False
        ret['message'] = 'Length of strings must be between 1 and 20'
        return ret
    if len(kwargs['characters']) >= 80:
        ret['res'] = False
        ret['message'] = 'Length of characters must be less than 80.'
        return ret
    if isinstance(api_version, int):
        api_version = str(api_version)
    if 'replacement' not in kwargs:
        replacement = True
    else:
        replacement = kwargs['replacement']
    _function = RANDOM_ORG_FUNCTIONS.get(api_version).get('generateStrings').get('method')
    data = {}
    data['id'] = 1911220
    data['jsonrpc'] = '2.0'
    data['method'] = _function
    data['params'] = {'apiKey': api_key, 'n': kwargs['number'], 'length': kwargs['length'], 'characters': kwargs['characters'], 'replacement': replacement}
    result = _query(api_version=api_version, data=data)
    if result:
        if 'random' in result:
            random_data = result.get('random').get('data')
            ret['data'] = random_data
        else:
            ret['res'] = False
            ret['message'] = result['message']
    else:
        ret['res'] = False
        ret['message'] = result['message']
    return ret

def generateUUIDs(api_key=None, api_version=None, **kwargs):
    if False:
        print('Hello World!')
    "\n    Generate a list of random UUIDs\n\n    :param api_key: The Random.org api key.\n    :param api_version: The Random.org api version.\n    :param number: How many random UUIDs you need.\n                   Must be within the [1,1e3] range.\n    :return: A list of UUIDs\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' random_org.generateUUIDs number=5\n\n    "
    ret = {'res': True}
    if not api_key or not api_version:
        try:
            options = __salt__['config.option']('random_org')
            if not api_key:
                api_key = options.get('api_key')
            if not api_version:
                api_version = options.get('api_version')
        except (NameError, KeyError, AttributeError):
            log.error('No Random.org api key found.')
            ret['message'] = 'No Random.org api key or api version found.'
            ret['res'] = False
            return ret
    for item in ['number']:
        if item not in kwargs:
            ret['res'] = False
            ret['message'] = 'Required argument, {} is missing.'.format(item)
            return ret
    if isinstance(api_version, int):
        api_version = str(api_version)
    if not _numeric(kwargs['number']) or not 1 <= kwargs['number'] <= 1000:
        ret['res'] = False
        ret['message'] = 'Number of UUIDs must be between 1 and 1000'
        return ret
    _function = RANDOM_ORG_FUNCTIONS.get(api_version).get('generateUUIDs').get('method')
    data = {}
    data['id'] = 1911220
    data['jsonrpc'] = '2.0'
    data['method'] = _function
    data['params'] = {'apiKey': api_key, 'n': kwargs['number']}
    result = _query(api_version=api_version, data=data)
    if result:
        if 'random' in result:
            random_data = result.get('random').get('data')
            ret['data'] = random_data
        else:
            ret['res'] = False
            ret['message'] = result['message']
    else:
        ret['res'] = False
        ret['message'] = result['message']
    return ret

def generateDecimalFractions(api_key=None, api_version=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Generates true random decimal fractions\n\n    :param api_key: The Random.org api key.\n    :param api_version: The Random.org api version.\n    :param number: How many random decimal fractions\n                   you need. Must be within the [1,1e4] range.\n    :param decimalPlaces: The number of decimal places\n                          to use. Must be within the [1,20] range.\n    :param replacement: Specifies whether the random numbers should\n                        be picked with replacement. The default (true)\n                        will cause the numbers to be picked with replacement,\n                        i.e., the resulting numbers may contain duplicate\n                        values (like a series of dice rolls). If you want the\n                        numbers picked to be unique (like raffle tickets drawn\n                        from a container), set this value to false.\n    :return: A list of decimal fraction\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' random_org.generateDecimalFractions number=10 decimalPlaces=4\n\n        salt '*' random_org.generateDecimalFractions number=10 decimalPlaces=4 replacement=True\n\n    "
    ret = {'res': True}
    if not api_key or not api_version:
        try:
            options = __salt__['config.option']('random_org')
            if not api_key:
                api_key = options.get('api_key')
            if not api_version:
                api_version = options.get('api_version')
        except (NameError, KeyError, AttributeError):
            log.error('No Random.org api key found.')
            ret['message'] = 'No Random.org api key or api version found.'
            ret['res'] = False
            return ret
    for item in ['number', 'decimalPlaces']:
        if item not in kwargs:
            ret['res'] = False
            ret['message'] = 'Required argument, {} is missing.'.format(item)
            return ret
    if not isinstance(kwargs['number'], int) or not 1 <= kwargs['number'] <= 10000:
        ret['res'] = False
        ret['message'] = 'Number of decimal fractions must be between 1 and 10000'
        return ret
    if not _numeric(kwargs['decimalPlaces']) or not 1 <= kwargs['decimalPlaces'] <= 20:
        ret['res'] = False
        ret['message'] = 'Number of decimal places must be between 1 and 20'
        return ret
    if 'replacement' not in kwargs:
        replacement = True
    else:
        replacement = kwargs['replacement']
    if isinstance(api_version, int):
        api_version = str(api_version)
    _function = RANDOM_ORG_FUNCTIONS.get(api_version).get('generateDecimalFractions').get('method')
    data = {}
    data['id'] = 1911220
    data['jsonrpc'] = '2.0'
    data['method'] = _function
    data['params'] = {'apiKey': api_key, 'n': kwargs['number'], 'decimalPlaces': kwargs['decimalPlaces'], 'replacement': replacement}
    result = _query(api_version=api_version, data=data)
    if result:
        if 'random' in result:
            random_data = result.get('random').get('data')
            ret['data'] = random_data
        else:
            ret['res'] = False
            ret['message'] = result['message']
    else:
        ret['res'] = False
        ret['message'] = result['message']
    return ret

def generateGaussians(api_key=None, api_version=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    This method generates true random numbers from a\n    Gaussian distribution (also known as a normal distribution).\n\n    :param api_key: The Random.org api key.\n    :param api_version: The Random.org api version.\n    :param number: How many random numbers you need.\n                   Must be within the [1,1e4] range.\n    :param mean: The distribution's mean. Must be\n                 within the [-1e6,1e6] range.\n    :param standardDeviation: The distribution's standard\n                              deviation. Must be within\n                              the [-1e6,1e6] range.\n    :param significantDigits: The number of significant digits\n                              to use. Must be within the [2,20] range.\n    :return: The user list.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' random_org.generateGaussians number=10 mean=0.0 standardDeviation=1.0 significantDigits=8\n\n    "
    ret = {'res': True}
    if not api_key or not api_version:
        try:
            options = __salt__['config.option']('random_org')
            if not api_key:
                api_key = options.get('api_key')
            if not api_version:
                api_version = options.get('api_version')
        except (NameError, KeyError, AttributeError):
            log.error('No Random.org api key found.')
            ret['message'] = 'No Random.org api key or api version found.'
            ret['res'] = False
            return ret
    for item in ['number', 'mean', 'standardDeviation', 'significantDigits']:
        if item not in kwargs:
            ret['res'] = False
            ret['message'] = 'Required argument, {} is missing.'.format(item)
            return ret
    if not _numeric(kwargs['number']) or not 1 <= kwargs['number'] <= 10000:
        ret['res'] = False
        ret['message'] = 'Number of decimal fractions must be between 1 and 10000'
        return ret
    if not _numeric(kwargs['mean']) or not -1000000 <= kwargs['mean'] <= 1000000:
        ret['res'] = False
        ret['message'] = "The distribution's mean must be between -1000000 and 1000000"
        return ret
    if not _numeric(kwargs['standardDeviation']) or not -1000000 <= kwargs['standardDeviation'] <= 1000000:
        ret['res'] = False
        ret['message'] = "The distribution's standard deviation must be between -1000000 and 1000000"
        return ret
    if not _numeric(kwargs['significantDigits']) or not 2 <= kwargs['significantDigits'] <= 20:
        ret['res'] = False
        ret['message'] = 'The number of significant digits must be between 2 and 20'
        return ret
    if isinstance(api_version, int):
        api_version = str(api_version)
    _function = RANDOM_ORG_FUNCTIONS.get(api_version).get('generateGaussians').get('method')
    data = {}
    data['id'] = 1911220
    data['jsonrpc'] = '2.0'
    data['method'] = _function
    data['params'] = {'apiKey': api_key, 'n': kwargs['number'], 'mean': kwargs['mean'], 'standardDeviation': kwargs['standardDeviation'], 'significantDigits': kwargs['significantDigits']}
    result = _query(api_version=api_version, data=data)
    if result:
        if 'random' in result:
            random_data = result.get('random').get('data')
            ret['data'] = random_data
        else:
            ret['res'] = False
            ret['message'] = result['message']
    else:
        ret['res'] = False
        ret['message'] = result['message']
    return ret

def generateBlobs(api_key=None, api_version=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    List all Slack users.\n\n    :param api_key: The Random.org api key.\n    :param api_version: The Random.org api version.\n    :param format: Specifies the format in which the\n                   blobs will be returned. Values\n                   allowed are base64 and hex.\n    :return: The user list.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' get_integers number=5 min=1 max=6\n\n        salt '*' get_integers number=5 min=1 max=6\n    "
    ret = {'res': True}
    if not api_key or not api_version:
        try:
            options = __salt__['config.option']('random_org')
            if not api_key:
                api_key = options.get('api_key')
            if not api_version:
                api_version = options.get('api_version')
        except (NameError, KeyError, AttributeError):
            log.error('No Random.org api key found.')
            ret['message'] = 'No Random.org api key or api version found.'
            ret['res'] = False
            return ret
    for item in ['number', 'size']:
        if item not in kwargs:
            ret['res'] = False
            ret['message'] = 'Required argument, {} is missing.'.format(item)
            return ret
    if not _numeric(kwargs['number']) or not 1 <= kwargs['number'] <= 100:
        ret['res'] = False
        ret['message'] = 'Number of blobs must be between 1 and 100'
        return ret
    if not _numeric(kwargs['size']) or not 1 <= kwargs['size'] <= 1048576 or kwargs['size'] % 8 != 0:
        ret['res'] = False
        ret['message'] = 'Number of blobs must be between 1 and 100'
        return ret
    if 'format' in kwargs:
        _format = kwargs['format']
        if _format not in ['base64', 'hex']:
            ret['res'] = False
            ret['message'] = 'Format must be either base64 or hex.'
            return ret
    else:
        _format = 'base64'
    if isinstance(api_version, int):
        api_version = str(api_version)
    _function = RANDOM_ORG_FUNCTIONS.get(api_version).get('generateBlobs').get('method')
    data = {}
    data['id'] = 1911220
    data['jsonrpc'] = '2.0'
    data['method'] = _function
    data['params'] = {'apiKey': api_key, 'n': kwargs['number'], 'size': kwargs['size'], 'format': _format}
    result = _query(api_version=api_version, data=data)
    if result:
        if 'random' in result:
            random_data = result.get('random').get('data')
            ret['data'] = random_data
        else:
            ret['res'] = False
            ret['message'] = result['message']
    else:
        ret['res'] = False
        ret['message'] = result['message']
    return ret