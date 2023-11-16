"""Uses Librosa to calculate the `bpm` field.
"""
from librosa import beat, load
from soundfile import LibsndfileError
from beets import ui, util
from beets.plugins import BeetsPlugin

class AutoBPMPlugin(BeetsPlugin):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.config.add({'auto': True, 'overwrite': False})
        if self.config['auto'].get(bool):
            self.import_stages = [self.imported]

    def commands(self):
        if False:
            while True:
                i = 10
        cmd = ui.Subcommand('autobpm', help='detect and add bpm from audio using Librosa')
        cmd.func = self.command
        return [cmd]

    def command(self, lib, opts, args):
        if False:
            print('Hello World!')
        self.calculate_bpm(lib.items(ui.decargs(args)), write=ui.should_write())

    def imported(self, session, task):
        if False:
            return 10
        self.calculate_bpm(task.imported_items())

    def calculate_bpm(self, items, write=False):
        if False:
            return 10
        overwrite = self.config['overwrite'].get(bool)
        for item in items:
            if item['bpm']:
                self._log.info('found bpm {0} for {1}', item['bpm'], util.displayable_path(item.path))
                if not overwrite:
                    continue
            try:
                (y, sr) = load(util.syspath(item.path), res_type='kaiser_fast')
            except LibsndfileError as exc:
                self._log.error('LibsndfileError: failed to load {0} {1}', util.displayable_path(item.path), exc)
                continue
            except ValueError as exc:
                self._log.error('ValueError: failed to load {0} {1}', util.displayable_path(item.path), exc)
                continue
            (tempo, _) = beat.beat_track(y=y, sr=sr)
            bpm = round(tempo)
            item['bpm'] = bpm
            self._log.info('added computed bpm {0} for {1}', bpm, util.displayable_path(item.path))
            if write:
                item.try_write()
            item.store()