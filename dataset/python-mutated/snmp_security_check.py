import bandit
from bandit.core import issue
from bandit.core import test_properties as test

@test.checks('Call')
@test.test_id('B508')
def snmp_insecure_version_check(context):
    if False:
        print('Hello World!')
    "**B508: Checking for insecure SNMP versions**\n\n    This test is for checking for the usage of insecure SNMP version like\n      v1, v2c\n\n    Please update your code to use more secure versions of SNMP.\n\n    :Example:\n\n    .. code-block:: none\n\n        >> Issue: [B508:snmp_insecure_version_check] The use of SNMPv1 and\n           SNMPv2 is insecure. You should use SNMPv3 if able.\n           Severity: Medium Confidence: High\n           CWE: CWE-319 (https://cwe.mitre.org/data/definitions/319.html)\n           Location: examples/snmp.py:4:4\n           More Info: https://bandit.readthedocs.io/en/latest/plugins/b508_snmp_insecure_version_check.html\n        3   # SHOULD FAIL\n        4   a = CommunityData('public', mpModel=0)\n        5   # SHOULD FAIL\n\n    .. seealso::\n\n     - http://snmplabs.com/pysnmp/examples/hlapi/asyncore/sync/manager/cmdgen/snmp-versions.html\n     - https://cwe.mitre.org/data/definitions/319.html\n\n    .. versionadded:: 1.7.2\n\n    .. versionchanged:: 1.7.3\n        CWE information added\n\n    "
    if context.call_function_name_qual == 'pysnmp.hlapi.CommunityData':
        if context.check_call_arg_value('mpModel', 0) or context.check_call_arg_value('mpModel', 1):
            return bandit.Issue(severity=bandit.MEDIUM, confidence=bandit.HIGH, cwe=issue.Cwe.CLEARTEXT_TRANSMISSION, text='The use of SNMPv1 and SNMPv2 is insecure. You should use SNMPv3 if able.', lineno=context.get_lineno_for_call_arg('CommunityData'))

@test.checks('Call')
@test.test_id('B509')
def snmp_crypto_check(context):
    if False:
        i = 10
        return i + 15
    '**B509: Checking for weak cryptography**\n\n    This test is for checking for the usage of insecure SNMP cryptography:\n      v3 using noAuthNoPriv.\n\n    Please update your code to use more secure versions of SNMP. For example:\n\n    Instead of:\n      `CommunityData(\'public\', mpModel=0)`\n\n    Use (Defaults to usmHMACMD5AuthProtocol and usmDESPrivProtocol\n      `UsmUserData("securityName", "authName", "privName")`\n\n    :Example:\n\n    .. code-block:: none\n\n        >> Issue: [B509:snmp_crypto_check] You should not use SNMPv3 without encryption. noAuthNoPriv & authNoPriv is insecure\n           Severity: Medium CWE: CWE-319 (https://cwe.mitre.org/data/definitions/319.html) Confidence: High\n           Location: examples/snmp.py:6:11\n           More Info: https://bandit.readthedocs.io/en/latest/plugins/b509_snmp_crypto_check.html\n        5   # SHOULD FAIL\n        6   insecure = UsmUserData("securityName")\n        7   # SHOULD FAIL\n\n    .. seealso::\n\n     - http://snmplabs.com/pysnmp/examples/hlapi/asyncore/sync/manager/cmdgen/snmp-versions.html\n     - https://cwe.mitre.org/data/definitions/319.html\n\n    .. versionadded:: 1.7.2\n\n    .. versionchanged:: 1.7.3\n        CWE information added\n\n    '
    if context.call_function_name_qual == 'pysnmp.hlapi.UsmUserData':
        if context.call_args_count < 3:
            return bandit.Issue(severity=bandit.MEDIUM, confidence=bandit.HIGH, cwe=issue.Cwe.CLEARTEXT_TRANSMISSION, text='You should not use SNMPv3 without encryption. noAuthNoPriv & authNoPriv is insecure', lineno=context.get_lineno_for_call_arg('UsmUserData'))