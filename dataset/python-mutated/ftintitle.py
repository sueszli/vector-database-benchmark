"""Moves "featured" artists to the title from the artist field.
"""
import re
from beets import plugins, ui
from beets.util import displayable_path

def split_on_feat(artist):
    if False:
        for i in range(10):
            print('nop')
    'Given an artist string, split the "main" artist from any artist\n    on the right-hand side of a string like "feat". Return the main\n    artist, which is always a string, and the featuring artist, which\n    may be a string or None if none is present.\n    '
    regex = re.compile(plugins.feat_tokens(), re.IGNORECASE)
    parts = [s.strip() for s in regex.split(artist, 1)]
    if len(parts) == 1:
        return (parts[0], None)
    else:
        return tuple(parts)

def contains_feat(title):
    if False:
        return 10
    'Determine whether the title contains a "featured" marker.'
    return bool(re.search(plugins.feat_tokens(), title, flags=re.IGNORECASE))

def find_feat_part(artist, albumartist):
    if False:
        print('Hello World!')
    "Attempt to find featured artists in the item's artist fields and\n    return the results. Returns None if no featured artist found.\n    "
    albumartist_split = artist.split(albumartist, 1)
    if len(albumartist_split) <= 1:
        return None
    elif albumartist_split[1] != '':
        (_, feat_part) = split_on_feat(albumartist_split[1])
        return feat_part
    else:
        (lhs, rhs) = split_on_feat(albumartist_split[0])
        if lhs:
            return lhs
    return None

class FtInTitlePlugin(plugins.BeetsPlugin):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.config.add({'auto': True, 'drop': False, 'format': 'feat. {0}'})
        self._command = ui.Subcommand('ftintitle', help='move featured artists to the title field')
        self._command.parser.add_option('-d', '--drop', dest='drop', action='store_true', default=None, help='drop featuring from artists and ignore title update')
        if self.config['auto']:
            self.import_stages = [self.imported]

    def commands(self):
        if False:
            print('Hello World!')

        def func(lib, opts, args):
            if False:
                print('Hello World!')
            self.config.set_args(opts)
            drop_feat = self.config['drop'].get(bool)
            write = ui.should_write()
            for item in lib.items(ui.decargs(args)):
                self.ft_in_title(item, drop_feat)
                item.store()
                if write:
                    item.try_write()
        self._command.func = func
        return [self._command]

    def imported(self, session, task):
        if False:
            for i in range(10):
                print('nop')
        'Import hook for moving featuring artist automatically.'
        drop_feat = self.config['drop'].get(bool)
        for item in task.imported_items():
            self.ft_in_title(item, drop_feat)
            item.store()

    def update_metadata(self, item, feat_part, drop_feat):
        if False:
            i = 10
            return i + 15
        'Choose how to add new artists to the title and set the new\n        metadata. Also, print out messages about any changes that are made.\n        If `drop_feat` is set, then do not add the artist to the title; just\n        remove it from the artist field.\n        '
        self._log.info('artist: {0} -> {1}', item.artist, item.albumartist)
        item.artist = item.albumartist
        if item.artist_sort:
            (item.artist_sort, _) = split_on_feat(item.artist_sort)
        if not drop_feat and (not contains_feat(item.title)):
            feat_format = self.config['format'].as_str()
            new_format = feat_format.format(feat_part)
            new_title = f'{item.title} {new_format}'
            self._log.info('title: {0} -> {1}', item.title, new_title)
            item.title = new_title

    def ft_in_title(self, item, drop_feat):
        if False:
            i = 10
            return i + 15
        "Look for featured artists in the item's artist fields and move\n        them to the title.\n        "
        artist = item.artist.strip()
        albumartist = item.albumartist.strip()
        (_, featured) = split_on_feat(artist)
        if featured and albumartist != artist and albumartist:
            self._log.info('{}', displayable_path(item.path))
            feat_part = None
            feat_part = find_feat_part(artist, albumartist)
            if feat_part:
                self.update_metadata(item, feat_part, drop_feat)
            else:
                self._log.info('no featuring artists found')