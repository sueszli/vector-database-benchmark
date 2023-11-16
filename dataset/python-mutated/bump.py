from trashcli.fs import read_file, write_file
import re

def version_from_date(today):
    if False:
        return 10
    return '0.%s.%s.%s' % (today.year % 100, today.month, today.day)

def save_new_version(new_version, path):
    if False:
        i = 10
        return i + 15
    content = read_file(path)
    new_content = re.sub('^version(\\s*)=.*', "version = '%s'" % new_version, content, flags=re.MULTILINE)
    write_file(path, new_content)