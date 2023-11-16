"""Update library's tags using MusicBrainz.
"""
import re
from collections import defaultdict
from beets import autotag, library, ui, util
from beets.autotag import hooks
from beets.plugins import BeetsPlugin, apply_item_changes
MBID_REGEX = '(\\d|\\w){8}-(\\d|\\w){4}-(\\d|\\w){4}-(\\d|\\w){4}-(\\d|\\w){12}'

class MBSyncPlugin(BeetsPlugin):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()

    def commands(self):
        if False:
            return 10
        cmd = ui.Subcommand('mbsync', help='update metadata from musicbrainz')
        cmd.parser.add_option('-p', '--pretend', action='store_true', help='show all changes but do nothing')
        cmd.parser.add_option('-m', '--move', action='store_true', dest='move', help='move files in the library directory')
        cmd.parser.add_option('-M', '--nomove', action='store_false', dest='move', help="don't move files in library")
        cmd.parser.add_option('-W', '--nowrite', action='store_false', default=None, dest='write', help="don't write updated metadata to files")
        cmd.parser.add_format_option()
        cmd.func = self.func
        return [cmd]

    def func(self, lib, opts, args):
        if False:
            print('Hello World!')
        'Command handler for the mbsync function.'
        move = ui.should_move(opts.move)
        pretend = opts.pretend
        write = ui.should_write(opts.write)
        query = ui.decargs(args)
        self.singletons(lib, query, move, pretend, write)
        self.albums(lib, query, move, pretend, write)

    def singletons(self, lib, query, move, pretend, write):
        if False:
            for i in range(10):
                print('nop')
        'Retrieve and apply info from the autotagger for items matched by\n        query.\n        '
        for item in lib.items(query + ['singleton:true']):
            item_formatted = format(item)
            if not item.mb_trackid:
                self._log.info('Skipping singleton with no mb_trackid: {0}', item_formatted)
                continue
            if not re.match(MBID_REGEX, item.mb_trackid):
                self._log.info('Skipping singleton with invalid mb_trackid:' + ' {0}', item_formatted)
                continue
            track_info = hooks.track_for_mbid(item.mb_trackid)
            if not track_info:
                self._log.info('Recording ID not found: {0} for track {0}', item.mb_trackid, item_formatted)
                continue
            with lib.transaction():
                autotag.apply_item_metadata(item, track_info)
                apply_item_changes(lib, item, move, pretend, write)

    def albums(self, lib, query, move, pretend, write):
        if False:
            for i in range(10):
                print('nop')
        'Retrieve and apply info from the autotagger for albums matched by\n        query and their items.\n        '
        for a in lib.albums(query):
            album_formatted = format(a)
            if not a.mb_albumid:
                self._log.info('Skipping album with no mb_albumid: {0}', album_formatted)
                continue
            items = list(a.items())
            if not re.match(MBID_REGEX, a.mb_albumid):
                self._log.info('Skipping album with invalid mb_albumid: {0}', album_formatted)
                continue
            album_info = hooks.album_for_mbid(a.mb_albumid)
            if not album_info:
                self._log.info('Release ID {0} not found for album {1}', a.mb_albumid, album_formatted)
                continue
            releasetrack_index = {}
            track_index = defaultdict(list)
            for track_info in album_info.tracks:
                releasetrack_index[track_info.release_track_id] = track_info
                track_index[track_info.track_id].append(track_info)
            mapping = {}
            for item in items:
                if item.mb_releasetrackid and item.mb_releasetrackid in releasetrack_index:
                    mapping[item] = releasetrack_index[item.mb_releasetrackid]
                else:
                    candidates = track_index[item.mb_trackid]
                    if len(candidates) == 1:
                        mapping[item] = candidates[0]
                    else:
                        for c in candidates:
                            if c.medium_index == item.track and c.medium == item.disc:
                                mapping[item] = c
                                break
            self._log.debug('applying changes to {}', album_formatted)
            with lib.transaction():
                autotag.apply_metadata(album_info, mapping)
                changed = False
                any_changed_item = items[0]
                for item in items:
                    item_changed = ui.show_model_changes(item)
                    changed |= item_changed
                    if item_changed:
                        any_changed_item = item
                        apply_item_changes(lib, item, move, pretend, write)
                if not changed:
                    continue
                if not pretend:
                    for key in library.Album.item_keys:
                        a[key] = any_changed_item[key]
                    a.store()
                    if move and lib.directory in util.ancestry(items[0].path):
                        self._log.debug('moving album {0}', album_formatted)
                        a.move()