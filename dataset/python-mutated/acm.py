"""
.. module: security_monkey.watchers.acm
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Alex Cline <alex.cline@gmail.com> @alex.cline

"""
from security_monkey.auditor import Auditor
from security_monkey.watchers.acm import ACM
from dateutil.tz import tzutc
from dateutil import parser

class ACMAuditor(Auditor):
    index = ACM.index
    i_am_singular = ACM.i_am_singular
    i_am_plural = ACM.i_am_plural

    def __init__(self, accounts=None, debug=False):
        if False:
            i = 10
            return i + 15
        super(ACMAuditor, self).__init__(accounts=accounts, debug=debug)

    def check_upcoming_expiration(self, cert_item):
        if False:
            return 10
        "\n        alert when a cert's expiration is within 30 days\n        "
        expiration = cert_item.config.get('NotAfter', None)
        if expiration:
            expiration = parser.parse(expiration)
            now = expiration.now(tzutc())
            time_to_expiration = (expiration - now).days
            if 0 <= time_to_expiration <= 30:
                notes = 'Expires on {0}.'.format(str(expiration))
                self.add_issue(10, 'Cert will expire soon.', cert_item, notes=notes)

    def check_future_expiration(self, cert_item):
        if False:
            print('Hello World!')
        "\n        alert when a cert's expiration is within 60 days\n        "
        expiration = cert_item.config.get('NotAfter', None)
        if expiration:
            expiration = parser.parse(expiration)
            now = expiration.now(tzutc())
            time_to_expiration = (expiration - now).days
            if 0 <= time_to_expiration <= 60:
                notes = 'Expires on {0}.'.format(str(expiration))
                self.add_issue(5, 'Cert will expire soon.', cert_item, notes=notes)

    def check_expired(self, cert_item):
        if False:
            print('Hello World!')
        '\n        alert when a cert is expired\n        '
        expiration = cert_item.config.get('NotAfter', None)
        if expiration:
            expiration = parser.parse(expiration)
            now = expiration.now(tzutc())
            time_to_expiration = (expiration - now).days
            if time_to_expiration < 0:
                notes = 'Expired on {0}.'.format(str(expiration))
                self.add_issue(10, 'Cert has expired.', cert_item, notes=notes)