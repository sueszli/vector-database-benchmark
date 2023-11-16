"""Determine BPM by pressing a key to the rhythm."""
import time
from beets import ui
from beets.plugins import BeetsPlugin

def bpm(max_strokes):
    if False:
        return 10
    'Returns average BPM (possibly of a playing song)\n    listening to Enter keystrokes.\n    '
    t0 = None
    dt = []
    for i in range(max_strokes):
        s = input()
        if s == '':
            t1 = time.time()
            if t0:
                dt.append(t1 - t0)
            t0 = t1
        else:
            break
    ave = sum([1.0 / dti * 60 for dti in dt]) / len(dt)
    return ave

class BPMPlugin(BeetsPlugin):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.config.add({'max_strokes': 3, 'overwrite': True})

    def commands(self):
        if False:
            return 10
        cmd = ui.Subcommand('bpm', help='determine bpm of a song by pressing a key to the rhythm')
        cmd.func = self.command
        return [cmd]

    def command(self, lib, opts, args):
        if False:
            return 10
        items = lib.items(ui.decargs(args))
        write = ui.should_write()
        self.get_bpm(items, write)

    def get_bpm(self, items, write=False):
        if False:
            while True:
                i = 10
        overwrite = self.config['overwrite'].get(bool)
        if len(items) > 1:
            raise ValueError('Can only get bpm of one song at time')
        item = items[0]
        if item['bpm']:
            self._log.info('Found bpm {0}', item['bpm'])
            if not overwrite:
                return
        self._log.info('Press Enter {0} times to the rhythm or Ctrl-D to exit', self.config['max_strokes'].get(int))
        new_bpm = bpm(self.config['max_strokes'].get(int))
        item['bpm'] = int(new_bpm)
        if write:
            item.try_write()
        item.store()
        self._log.info('Added new bpm {0}', item['bpm'])