import bandit
from bandit.core import issue
from bandit.core import test_properties as test

def get_bad_proto_versions(config):
    if False:
        print('Hello World!')
    return config['bad_protocol_versions']

def gen_config(name):
    if False:
        i = 10
        return i + 15
    if name == 'ssl_with_bad_version':
        return {'bad_protocol_versions': ['PROTOCOL_SSLv2', 'SSLv2_METHOD', 'SSLv23_METHOD', 'PROTOCOL_SSLv3', 'PROTOCOL_TLSv1', 'SSLv3_METHOD', 'TLSv1_METHOD', 'PROTOCOL_TLSv1_1', 'TLSv1_1_METHOD']}

@test.takes_config
@test.checks('Call')
@test.test_id('B502')
def ssl_with_bad_version(context, config):
    if False:
        for i in range(10):
            print('nop')
    '**B502: Test for SSL use with bad version used**\n\n    Several highly publicized exploitable flaws have been discovered\n    in all versions of SSL and early versions of TLS. It is strongly\n    recommended that use of the following known broken protocol versions be\n    avoided:\n\n    - SSL v2\n    - SSL v3\n    - TLS v1\n    - TLS v1.1\n\n    This plugin test scans for calls to Python methods with parameters that\n    indicate the used broken SSL/TLS protocol versions. Currently, detection\n    supports methods using Python\'s native SSL/TLS support and the pyOpenSSL\n    module. A HIGH severity warning will be reported whenever known broken\n    protocol versions are detected.\n\n    It is worth noting that native support for TLS 1.2 is only available in\n    more recent Python versions, specifically 2.7.9 and up, and 3.x\n\n    A note on \'SSLv23\':\n\n    Amongst the available SSL/TLS versions provided by Python/pyOpenSSL there\n    exists the option to use SSLv23. This very poorly named option actually\n    means "use the highest version of SSL/TLS supported by both the server and\n    client". This may (and should be) a version well in advance of SSL v2 or\n    v3. Bandit can scan for the use of SSLv23 if desired, but its detection\n    does not necessarily indicate a problem.\n\n    When using SSLv23 it is important to also provide flags to explicitly\n    exclude bad versions of SSL/TLS from the protocol versions considered. Both\n    the Python native and pyOpenSSL modules provide the ``OP_NO_SSLv2`` and\n    ``OP_NO_SSLv3`` flags for this purpose.\n\n    **Config Options:**\n\n    .. code-block:: yaml\n\n        ssl_with_bad_version:\n            bad_protocol_versions:\n                - PROTOCOL_SSLv2\n                - SSLv2_METHOD\n                - SSLv23_METHOD\n                - PROTOCOL_SSLv3  # strict option\n                - PROTOCOL_TLSv1  # strict option\n                - SSLv3_METHOD    # strict option\n                - TLSv1_METHOD    # strict option\n\n    :Example:\n\n    .. code-block:: none\n\n        >> Issue: ssl.wrap_socket call with insecure SSL/TLS protocol version\n        identified, security issue.\n           Severity: High   Confidence: High\n           CWE: CWE-327 (https://cwe.mitre.org/data/definitions/327.html)\n           Location: ./examples/ssl-insecure-version.py:13\n        12  # strict tests\n        13  ssl.wrap_socket(ssl_version=ssl.PROTOCOL_SSLv3)\n        14  ssl.wrap_socket(ssl_version=ssl.PROTOCOL_TLSv1)\n\n    .. seealso::\n\n     - :func:`ssl_with_bad_defaults`\n     - :func:`ssl_with_no_version`\n     - https://heartbleed.com/\n     - https://en.wikipedia.org/wiki/POODLE\n     - https://security.openstack.org/guidelines/dg_move-data-securely.html\n     - https://cwe.mitre.org/data/definitions/327.html\n\n    .. versionadded:: 0.9.0\n\n    .. versionchanged:: 1.7.3\n        CWE information added\n\n    .. versionchanged:: 1.7.5\n        Added TLS 1.1\n\n    '
    bad_ssl_versions = get_bad_proto_versions(config)
    if context.call_function_name_qual == 'ssl.wrap_socket':
        if context.check_call_arg_value('ssl_version', bad_ssl_versions):
            return bandit.Issue(severity=bandit.HIGH, confidence=bandit.HIGH, cwe=issue.Cwe.BROKEN_CRYPTO, text='ssl.wrap_socket call with insecure SSL/TLS protocol version identified, security issue.', lineno=context.get_lineno_for_call_arg('ssl_version'))
    elif context.call_function_name_qual == 'pyOpenSSL.SSL.Context':
        if context.check_call_arg_value('method', bad_ssl_versions):
            return bandit.Issue(severity=bandit.HIGH, confidence=bandit.HIGH, cwe=issue.Cwe.BROKEN_CRYPTO, text='SSL.Context call with insecure SSL/TLS protocol version identified, security issue.', lineno=context.get_lineno_for_call_arg('method'))
    elif context.call_function_name_qual != 'ssl.wrap_socket' and context.call_function_name_qual != 'pyOpenSSL.SSL.Context':
        if context.check_call_arg_value('method', bad_ssl_versions) or context.check_call_arg_value('ssl_version', bad_ssl_versions):
            lineno = context.get_lineno_for_call_arg('method') or context.get_lineno_for_call_arg('ssl_version')
            return bandit.Issue(severity=bandit.MEDIUM, confidence=bandit.MEDIUM, cwe=issue.Cwe.BROKEN_CRYPTO, text='Function call with insecure SSL/TLS protocol identified, possible security issue.', lineno=lineno)

@test.takes_config('ssl_with_bad_version')
@test.checks('FunctionDef')
@test.test_id('B503')
def ssl_with_bad_defaults(context, config):
    if False:
        for i in range(10):
            print('nop')
    "**B503: Test for SSL use with bad defaults specified**\n\n    This plugin is part of a family of tests that detect the use of known bad\n    versions of SSL/TLS, please see :doc:`../plugins/ssl_with_bad_version` for\n    a complete discussion. Specifically, this plugin test scans for Python\n    methods with default parameter values that specify the use of broken\n    SSL/TLS protocol versions. Currently, detection supports methods using\n    Python's native SSL/TLS support and the pyOpenSSL module. A MEDIUM severity\n    warning will be reported whenever known broken protocol versions are\n    detected.\n\n    **Config Options:**\n\n    This test shares the configuration provided for the standard\n    :doc:`../plugins/ssl_with_bad_version` test, please refer to its\n    documentation.\n\n    :Example:\n\n    .. code-block:: none\n\n        >> Issue: Function definition identified with insecure SSL/TLS protocol\n        version by default, possible security issue.\n           Severity: Medium   Confidence: Medium\n           CWE: CWE-327 (https://cwe.mitre.org/data/definitions/327.html)\n           Location: ./examples/ssl-insecure-version.py:28\n        27\n        28  def open_ssl_socket(version=SSL.SSLv2_METHOD):\n        29      pass\n\n    .. seealso::\n\n     - :func:`ssl_with_bad_version`\n     - :func:`ssl_with_no_version`\n     - https://heartbleed.com/\n     - https://en.wikipedia.org/wiki/POODLE\n     - https://security.openstack.org/guidelines/dg_move-data-securely.html\n\n    .. versionadded:: 0.9.0\n\n    .. versionchanged:: 1.7.3\n        CWE information added\n\n    .. versionchanged:: 1.7.5\n        Added TLS 1.1\n\n    "
    bad_ssl_versions = get_bad_proto_versions(config)
    for default in context.function_def_defaults_qual:
        val = default.split('.')[-1]
        if val in bad_ssl_versions:
            return bandit.Issue(severity=bandit.MEDIUM, confidence=bandit.MEDIUM, cwe=issue.Cwe.BROKEN_CRYPTO, text='Function definition identified with insecure SSL/TLS protocol version by default, possible security issue.')

@test.checks('Call')
@test.test_id('B504')
def ssl_with_no_version(context):
    if False:
        for i in range(10):
            print('nop')
    "**B504: Test for SSL use with no version specified**\n\n    This plugin is part of a family of tests that detect the use of known bad\n    versions of SSL/TLS, please see :doc:`../plugins/ssl_with_bad_version` for\n    a complete discussion. Specifically, This plugin test scans for specific\n    methods in Python's native SSL/TLS support and the pyOpenSSL module that\n    configure the version of SSL/TLS protocol to use. These methods are known\n    to provide default value that maximize compatibility, but permit use of the\n    aforementioned broken protocol versions. A LOW severity warning will be\n    reported whenever this is detected.\n\n    **Config Options:**\n\n    This test shares the configuration provided for the standard\n    :doc:`../plugins/ssl_with_bad_version` test, please refer to its\n    documentation.\n\n    :Example:\n\n    .. code-block:: none\n\n        >> Issue: ssl.wrap_socket call with no SSL/TLS protocol version\n        specified, the default SSLv23 could be insecure, possible security\n        issue.\n           Severity: Low   Confidence: Medium\n           CWE: CWE-327 (https://cwe.mitre.org/data/definitions/327.html)\n           Location: ./examples/ssl-insecure-version.py:23\n        22\n        23  ssl.wrap_socket()\n        24\n\n    .. seealso::\n\n     - :func:`ssl_with_bad_version`\n     - :func:`ssl_with_bad_defaults`\n     - https://heartbleed.com/\n     - https://en.wikipedia.org/wiki/POODLE\n     - https://security.openstack.org/guidelines/dg_move-data-securely.html\n\n    .. versionadded:: 0.9.0\n\n    .. versionchanged:: 1.7.3\n        CWE information added\n\n    "
    if context.call_function_name_qual == 'ssl.wrap_socket':
        if context.check_call_arg_value('ssl_version') is None:
            return bandit.Issue(severity=bandit.LOW, confidence=bandit.MEDIUM, cwe=issue.Cwe.BROKEN_CRYPTO, text='ssl.wrap_socket call with no SSL/TLS protocol version specified, the default SSLv23 could be insecure, possible security issue.', lineno=context.get_lineno_for_call_arg('ssl_version'))