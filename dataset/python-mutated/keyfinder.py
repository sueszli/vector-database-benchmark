"""Uses the `KeyFinder` program to add the `initial_key` field.
"""
import os.path
import subprocess
from beets import ui, util
from beets.plugins import BeetsPlugin

class KeyFinderPlugin(BeetsPlugin):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.config.add({'bin': 'KeyFinder', 'auto': True, 'overwrite': False})
        if self.config['auto'].get(bool):
            self.import_stages = [self.imported]

    def commands(self):
        if False:
            for i in range(10):
                print('nop')
        cmd = ui.Subcommand('keyfinder', help='detect and add initial key from audio')
        cmd.func = self.command
        return [cmd]

    def command(self, lib, opts, args):
        if False:
            for i in range(10):
                print('nop')
        self.find_key(lib.items(ui.decargs(args)), write=ui.should_write())

    def imported(self, session, task):
        if False:
            print('Hello World!')
        self.find_key(task.imported_items())

    def find_key(self, items, write=False):
        if False:
            print('Hello World!')
        overwrite = self.config['overwrite'].get(bool)
        command = [self.config['bin'].as_str()]
        if 'keyfinder-cli' not in os.path.basename(command[0]).lower():
            command.append('-f')
        for item in items:
            if item['initial_key'] and (not overwrite):
                continue
            try:
                output = util.command_output(command + [util.syspath(item.path)]).stdout
            except (subprocess.CalledProcessError, OSError) as exc:
                self._log.error('execution failed: {0}', exc)
                continue
            try:
                key_raw = output.rsplit(None, 1)[-1]
            except IndexError:
                self._log.error('no key returned for path: {0}', item.path)
                continue
            try:
                key = key_raw.decode('utf-8')
            except UnicodeDecodeError:
                self._log.error('output is invalid UTF-8')
                continue
            item['initial_key'] = key
            self._log.info('added computed initial key {0} for {1}', key, util.displayable_path(item.path))
            if write:
                item.try_write()
            item.store()