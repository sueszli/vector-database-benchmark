import re
from collections import namedtuple
_giturlmatcher = re.compile('(?P<proto>(https?://|ssh://|git://|))((?P<user>.*)@)?(?P<domain>[^\\/:]+)(:((?P<port>[0-9]+)/)?|/)((?P<owner>.+)/)?(?P<repo>[^/]+?)(\\.git)?$')
GitUrl = namedtuple('GitUrl', ['proto', 'user', 'domain', 'port', 'owner', 'repo'])

def giturlparse(url):
    if False:
        for i in range(10):
            print('nop')
    res = _giturlmatcher.match(url)
    if res is None:
        return None
    port = res.group('port')
    if port is not None:
        port = int(port)
    proto = res.group('proto')
    if proto:
        proto = proto[:-3]
    else:
        proto = 'ssh'
    return GitUrl(proto, res.group('user'), res.group('domain'), port, res.group('owner'), res.group('repo'))