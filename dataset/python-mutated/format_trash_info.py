import datetime
from six.moves.urllib.parse import quote as url_quote

def format_trashinfo(original_location, deletion_date):
    if False:
        return 10
    content = ('[Trash Info]\n' + 'Path=%s\n' % format_original_location(original_location) + 'DeletionDate=%s\n' % format_date(deletion_date)).encode('utf-8')
    return content

def format_date(deletion_date):
    if False:
        for i in range(10):
            print('nop')
    return deletion_date.strftime('%Y-%m-%dT%H:%M:%S')

def format_original_location(original_location):
    if False:
        i = 10
        return i + 15
    return url_quote(original_location, '/')