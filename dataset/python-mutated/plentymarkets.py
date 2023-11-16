from json import dumps, loads
import re
from oauthlib.common import to_unicode

def plentymarkets_compliance_fix(session):
    if False:
        for i in range(10):
            print('nop')

    def _to_snake_case(n):
        if False:
            for i in range(10):
                print('nop')
        return re.sub('(.)([A-Z][a-z]+)', '\\1_\\2', n).lower()

    def _compliance_fix(r):
        if False:
            print('Hello World!')
        if 'application/json' in r.headers.get('content-type', {}) and r.status_code == 200:
            token = loads(r.text)
        else:
            return r
        fixed_token = {}
        for (k, v) in token.items():
            fixed_token[_to_snake_case(k)] = v
        r._content = to_unicode(dumps(fixed_token)).encode('UTF-8')
        return r
    session.register_compliance_hook('access_token_response', _compliance_fix)
    return session