"""These are the default responses"""

def malicious_detector_response(observable: str, malicious: bool, timeout: bool=False) -> dict:
    if False:
        i = 10
        return i + 15
    'Standard response for malicious detector analyzers\n\n    :param observable: observable analyzed\n    :type observable: str\n    :param malicious: tell if the observable is reported as malicious from analyzer\n    :type malicious: bool\n    :param timeout: set if the DNS query timed-out\n    :type timeout bool\n    :return:\n    :rtype: dict\n    '
    report = {'observable': observable, 'malicious': malicious}
    if timeout:
        report['timeout'] = True
    return report

def dns_resolver_response(observable: str, resolutions: list=None, timeout: bool=False) -> dict:
    if False:
        for i in range(10):
            print('nop')
    'Standard response for DNS resolver analyzers\n\n    :param observable: observable analyzed\n    :type observable: str\n    :param resolutions: list of DNS resolutions, it is empty in case of no resolutions,\n    default to None\n    :type resolutions: list, optional\n    :param timeout: set if the DNS query timed-out\n    :type timeout bool\n    :return:\n    :rtype: dict\n    '
    if not resolutions:
        resolutions = []
    report = {'observable': observable, 'resolutions': resolutions}
    if timeout:
        report['timeout'] = True
    return report