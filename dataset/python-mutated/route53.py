from .. import AWSProperty
from . import boolean

class AliasTarget(AWSProperty):
    """
    Export:
    """
    props = {'HostedZoneId': (str, True), 'DNSName': (str, True), 'EvaluateTargetHealth': (boolean, False)}

    def __init__(self, hostedzoneid=None, dnsname=None, evaluatetargethealth=None, **kwargs):
        if False:
            print('Hello World!')
        if hostedzoneid is not None:
            kwargs['HostedZoneId'] = hostedzoneid
        if dnsname is not None:
            kwargs['DNSName'] = dnsname
        if evaluatetargethealth is not None:
            kwargs['EvaluateTargetHealth'] = evaluatetargethealth
        super().__init__(**kwargs)