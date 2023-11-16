"""Update library's tags using Beatport.
"""
from beets import autotag, library, ui, util
from beets.plugins import BeetsPlugin, apply_item_changes
from .beatport import BeatportPlugin

class BPSyncPlugin(BeetsPlugin):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.beatport_plugin = BeatportPlugin()
        self.beatport_plugin.setup()

    def commands(self):
        if False:
            print('Hello World!')
        cmd = ui.Subcommand('bpsync', help='update metadata from Beatport')
        cmd.parser.add_option('-p', '--pretend', action='store_true', help='show all changes but do nothing')
        cmd.parser.add_option('-m', '--move', action='store_true', dest='move', help='move files in the library directory')
        cmd.parser.add_option('-M', '--nomove', action='store_false', dest='move', help="don't move files in library")
        cmd.parser.add_option('-W', '--nowrite', action='store_false', default=None, dest='write', help="don't write updated metadata to files")
        cmd.parser.add_format_option()
        cmd.func = self.func
        return [cmd]

    def func(self, lib, opts, args):
        if False:
            i = 10
            return i + 15
        'Command handler for the bpsync function.'
        move = ui.should_move(opts.move)
        pretend = opts.pretend
        write = ui.should_write(opts.write)
        query = ui.decargs(args)
        self.singletons(lib, query, move, pretend, write)
        self.albums(lib, query, move, pretend, write)

    def singletons(self, lib, query, move, pretend, write):
        if False:
            while True:
                i = 10
        'Retrieve and apply info from the autotagger for items matched by\n        query.\n        '
        for item in lib.items(query + ['singleton:true']):
            if not item.mb_trackid:
                self._log.info('Skipping singleton with no mb_trackid: {}', item)
                continue
            if not self.is_beatport_track(item):
                self._log.info('Skipping non-{} singleton: {}', self.beatport_plugin.data_source, item)
                continue
            trackinfo = self.beatport_plugin.track_for_id(item.mb_trackid)
            with lib.transaction():
                autotag.apply_item_metadata(item, trackinfo)
                apply_item_changes(lib, item, move, pretend, write)

    @staticmethod
    def is_beatport_track(item):
        if False:
            while True:
                i = 10
        return item.get('data_source') == BeatportPlugin.data_source and item.mb_trackid.isnumeric()

    def get_album_tracks(self, album):
        if False:
            return 10
        if not album.mb_albumid:
            self._log.info('Skipping album with no mb_albumid: {}', album)
            return False
        if not album.mb_albumid.isnumeric():
            self._log.info('Skipping album with invalid {} ID: {}', self.beatport_plugin.data_source, album)
            return False
        items = list(album.items())
        if album.get('data_source') == self.beatport_plugin.data_source:
            return items
        if not all((self.is_beatport_track(item) for item in items)):
            self._log.info('Skipping non-{} release: {}', self.beatport_plugin.data_source, album)
            return False
        return items

    def albums(self, lib, query, move, pretend, write):
        if False:
            while True:
                i = 10
        'Retrieve and apply info from the autotagger for albums matched by\n        query and their items.\n        '
        for album in lib.albums(query):
            items = self.get_album_tracks(album)
            if not items:
                continue
            albuminfo = self.beatport_plugin.album_for_id(album.mb_albumid)
            if not albuminfo:
                self._log.info('Release ID {} not found for album {}', album.mb_albumid, album)
                continue
            beatport_trackid_to_trackinfo = {track.track_id: track for track in albuminfo.tracks}
            library_trackid_to_item = {int(item.mb_trackid): item for item in items}
            item_to_trackinfo = {item: beatport_trackid_to_trackinfo[track_id] for (track_id, item) in library_trackid_to_item.items()}
            self._log.info('applying changes to {}', album)
            with lib.transaction():
                autotag.apply_metadata(albuminfo, item_to_trackinfo)
                changed = False
                any_changed_item = items[0]
                for item in items:
                    item_changed = ui.show_model_changes(item)
                    changed |= item_changed
                    if item_changed:
                        any_changed_item = item
                        apply_item_changes(lib, item, move, pretend, write)
                if pretend or not changed:
                    continue
                for key in library.Album.item_keys:
                    album[key] = any_changed_item[key]
                album.store()
                if move and lib.directory in util.ancestry(items[0].path):
                    self._log.debug('moving album {}', album)
                    album.move()