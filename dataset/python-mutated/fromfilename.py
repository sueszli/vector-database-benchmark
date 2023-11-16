"""If the title is empty, try to extract track and title from the
filename.
"""
import os
import re
from beets import plugins
from beets.util import displayable_path
PATTERNS = ['^(?P<artist>.+)[\\-_](?P<title>.+)[\\-_](?P<tag>.*)$', '^(?P<track>\\d+)[\\s.\\-_]+(?P<artist>.+)[\\-_](?P<title>.+)[\\-_](?P<tag>.*)$', '^(?P<artist>.+)[\\-_](?P<title>.+)$', '^(?P<track>\\d+)[\\s.\\-_]+(?P<artist>.+)[\\-_](?P<title>.+)$', '^(?P<track>\\d+)[\\s.\\-_]+(?P<title>.+)$', '^(?P<track>\\d+)\\s+(?P<title>.+)$', '^(?P<title>.+) by (?P<artist>.+)$', '^(?P<track>\\d+).*$', '^(?P<title>.+)$']
BAD_TITLE_PATTERNS = ['^$']

def equal(seq):
    if False:
        i = 10
        return i + 15
    'Determine whether a sequence holds identical elements.'
    return len(set(seq)) <= 1

def equal_fields(matchdict, field):
    if False:
        while True:
            i = 10
    'Do all items in `matchdict`, whose values are dictionaries, have\n    the same value for `field`? (If they do, the field is probably not\n    the title.)\n    '
    return equal((m[field] for m in matchdict.values()))

def all_matches(names, pattern):
    if False:
        return 10
    'If all the filenames in the item/filename mapping match the\n    pattern, return a dictionary mapping the items to dictionaries\n    giving the value for each named subpattern in the match. Otherwise,\n    return None.\n    '
    matches = {}
    for (item, name) in names.items():
        m = re.match(pattern, name, re.IGNORECASE)
        if m and m.groupdict():
            matches[item] = m.groupdict()
        else:
            return None
    return matches

def bad_title(title):
    if False:
        print('Hello World!')
    'Determine whether a given title is "bad" (empty or otherwise\n    meaningless) and in need of replacement.\n    '
    for pat in BAD_TITLE_PATTERNS:
        if re.match(pat, title, re.IGNORECASE):
            return True
    return False

def apply_matches(d, log):
    if False:
        print('Hello World!')
    'Given a mapping from items to field dicts, apply the fields to\n    the objects.\n    '
    some_map = list(d.values())[0]
    keys = some_map.keys()
    if 'tag' in keys and (not equal_fields(d, 'tag')):
        return
    if 'artist' in keys:
        if equal_fields(d, 'artist'):
            artist = some_map['artist']
            title_field = 'title'
        elif equal_fields(d, 'title'):
            artist = some_map['title']
            title_field = 'artist'
        else:
            return
        for item in d:
            if not item.artist:
                item.artist = artist
                log.info('Artist replaced with: {}'.format(item.artist))
    else:
        title_field = 'title'
    for item in d:
        if bad_title(item.title):
            item.title = str(d[item][title_field])
            log.info('Title replaced with: {}'.format(item.title))
        if 'track' in d[item] and item.track == 0:
            item.track = int(d[item]['track'])
            log.info('Track replaced with: {}'.format(item.track))

class FromFilenamePlugin(plugins.BeetsPlugin):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.register_listener('import_task_start', self.filename_task)

    def filename_task(self, task, session):
        if False:
            while True:
                i = 10
        'Examine each item in the task to see if we can extract a title\n        from the filename. Try to match all filenames to a number of\n        regexps, starting with the most complex patterns and successively\n        trying less complex patterns. As soon as all filenames match the\n        same regex we can make an educated guess of which part of the\n        regex that contains the title.\n        '
        items = task.items if task.is_album else [task.item]
        missing_titles = sum((bad_title(i.title) for i in items))
        if missing_titles:
            names = {}
            for item in items:
                path = displayable_path(item.path)
                (name, _) = os.path.splitext(os.path.basename(path))
                names[item] = name
            for pattern in PATTERNS:
                d = all_matches(names, pattern)
                if d:
                    apply_matches(d, self._log)