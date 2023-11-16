from datetime import datetime, timedelta
from scylla.database import ProxyIP

class ValidationPolicy(object):
    """
    ValidationPolicy will make decision about validating a proxy IP from the following aspects:
    1. Whether or not to validate the proxy
    2. Use http or https to validate the proxy

    After 3 attempts, the validator should try no more attempts in 24 hours after its creation.
    """
    proxy_ip: ProxyIP = None

    def __init__(self, proxy_ip: ProxyIP):
        if False:
            for i in range(10):
                print('nop')
        '\n        Constructor of ValidationPolicy\n        :param proxy_ip: the ProxyIP instance to be validated\n        '
        self.proxy_ip = proxy_ip

    def should_validate(self) -> bool:
        if False:
            i = 10
            return i + 15
        if self.proxy_ip.attempts == 0:
            return True
        elif self.proxy_ip.attempts < 3 and datetime.now() - self.proxy_ip.created_at < timedelta(hours=24) and (not self.proxy_ip.is_valid):
            return True
        elif timedelta(hours=48) > datetime.now() - self.proxy_ip.created_at > timedelta(hours=24) and self.proxy_ip.attempts < 6:
            return True
        elif datetime.now() - self.proxy_ip.created_at < timedelta(days=7) and self.proxy_ip.attempts < 21 and self.proxy_ip.is_valid:
            return True
        return False

    def should_try_https(self) -> bool:
        if False:
            print('Hello World!')
        if self.proxy_ip.is_valid and self.proxy_ip.attempts < 3 and (self.proxy_ip.https_attempts == 0):
            return True
        return False