import getpass
import re

def parse(uri, user=None, port=22):
    if False:
        print('Hello World!')
    '\n    parses ssh connection uri-like sentences.\n    ex:\n        - root@google.com -> (root, google.com, 22)\n        - noreply@facebook.com:22 -> (noreply, facebook.com, 22)\n        - facebook.com:3306 -> ($USER, facebook.com, 3306)\n        - twitter.com -> ($USER, twitter.com, 22)\n\n    default port: 22\n    default user: $USER (getpass.getuser())\n    '
    uri = uri.strip()
    if not user:
        user = getpass.getuser()
    if '@' in uri:
        user = uri.split('@')[0]
    if ':' in uri:
        port = uri.split(':')[-1]
    try:
        port = int(port)
    except ValueError:
        raise ValueError('port must be numeric.')
    uri = re.sub(':.*', '', uri)
    uri = re.sub('.*@', '', uri)
    host = uri
    return (user, host, port)