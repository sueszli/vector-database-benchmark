import re
import urllib
from typing import Optional
_GITBLOB_RE = re.compile('(?P<base>https:\\/\\/?(gist\\.)?github.com\\/)(?P<account>([\\w\\.]+\\/){1,2})(?P<blob_or_raw>(blob|raw))?(?P<suffix>(.+)?)')

def process_gitblob_url(url: str) -> str:
    if False:
        while True:
            i = 10
    'Check url to see if it describes a GitHub Gist "blob" URL.\n\n    If so, returns a new URL to get the "raw" script.\n    If not, returns URL unchanged.\n    '
    match = _GITBLOB_RE.match(url)
    if match:
        mdict = match.groupdict()
        if mdict['blob_or_raw'] == 'blob':
            return '{base}{account}raw{suffix}'.format(**mdict)
        if mdict['blob_or_raw'] == 'raw':
            return url
        return url + '/raw'
    return url

def get_hostname(url: str) -> Optional[str]:
    if False:
        return 10
    'Return the hostname of a URL (with or without protocol).'
    if '://' not in url:
        url = 'http://%s' % url
    parsed = urllib.parse.urlparse(url)
    return parsed.hostname

def print_url(title, url):
    if False:
        i = 10
        return i + 15
    'Pretty-print a URL on the terminal.'
    import click
    click.secho('  %s: ' % title, nl=False, fg='blue')
    click.secho(url, bold=True)