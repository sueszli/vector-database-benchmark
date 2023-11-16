import sys
import win32security
from sspi import ClientAuth, ServerAuth

def validate(username, password, domain=''):
    if False:
        return 10
    auth_info = (username, domain, password)
    ca = ClientAuth('NTLM', auth_info=auth_info)
    sa = ServerAuth('NTLM')
    data = err = None
    while err != 0:
        (err, data) = ca.authorize(data)
        (err, data) = sa.authorize(data)
if __name__ == '__main__':
    if len(sys.argv) not in [2, 3, 4]:
        print(f'Usage: {__file__} username [password [domain]]')
        sys.exit(1)
    password = None
    if len(sys.argv) >= 3:
        password = sys.argv[2]
    domain = ''
    if len(sys.argv) >= 4:
        domain = sys.argv[3]
    try:
        validate(sys.argv[1], password, domain)
        print('Validated OK')
    except win32security.error as details:
        (hr, func, msg) = details
        print('Validation failed: %s (%d)' % (msg, hr))