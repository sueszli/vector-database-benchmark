import re
import warnings
from six.moves import urllib_parse

def parse_url(url, warning=True):
    if False:
        while True:
            i = 10
    'Parse URLs especially for Google Drive links.\n\n    file_id: ID of file on Google Drive.\n    is_download_link: Flag if it is download link of Google Drive.\n    '
    parsed = urllib_parse.urlparse(url)
    query = urllib_parse.parse_qs(parsed.query)
    is_gdrive = parsed.hostname in ['drive.google.com', 'docs.google.com']
    is_download_link = parsed.path.endswith('/uc')
    if not is_gdrive:
        return (is_gdrive, is_download_link)
    file_id = None
    if 'id' in query:
        file_ids = query['id']
        if len(file_ids) == 1:
            file_id = file_ids[0]
    else:
        patterns = ['^/file/d/(.*?)/(edit|view)$', '^/file/u/[0-9]+/d/(.*?)/(edit|view)$', '^/document/d/(.*?)/(edit|htmlview|view)$', '^/document/u/[0-9]+/d/(.*?)/(edit|htmlview|view)$', '^/presentation/d/(.*?)/(edit|htmlview|view)$', '^/presentation/u/[0-9]+/d/(.*?)/(edit|htmlview|view)$', '^/spreadsheets/d/(.*?)/(edit|htmlview|view)$', '^/spreadsheets/u/[0-9]+/d/(.*?)/(edit|htmlview|view)$']
        for pattern in patterns:
            match = re.match(pattern, parsed.path)
            if match:
                file_id = match.groups()[0]
                break
    if warning and (not is_download_link):
        warnings.warn('You specified a Google Drive link that is not the correct link to download a file. You might want to try `--fuzzy` option or the following url: {url}'.format(url='https://drive.google.com/uc?id={}'.format(file_id)))
    return (file_id, is_download_link)