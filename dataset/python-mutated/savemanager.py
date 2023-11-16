"""Saving things to disk periodically."""
import os.path
import collections
from typing import MutableMapping
from qutebrowser.qt.core import pyqtSlot, QObject, QTimer
from qutebrowser.config import config
from qutebrowser.api import cmdutils
from qutebrowser.utils import utils, log, message, usertypes, error
from qutebrowser.misc import objects

class Saveable:
    """A single thing which can be saved.

    Attributes:
        _name: The name of the thing to be saved.
        _dirty: Whether the saveable was changed since the last save.
        _save_handler: The function to call to save this Saveable.
        _save_on_exit: Whether to always save this saveable on exit.
        _config_opt: A config option which decides whether to auto-save or not.
                     None if no such option exists.
        _filename: The filename of the underlying file.
    """

    def __init__(self, name, save_handler, changed=None, config_opt=None, filename=None):
        if False:
            i = 10
            return i + 15
        self._name = name
        self._dirty = False
        self._save_handler = save_handler
        self._config_opt = config_opt
        if changed is not None:
            changed.connect(self.mark_dirty)
            self._save_on_exit = False
        else:
            self._save_on_exit = True
        self._filename = filename
        if filename is not None and (not os.path.exists(filename)):
            self._dirty = True
            self.save()

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return utils.get_repr(self, name=self._name, dirty=self._dirty, save_handler=self._save_handler, config_opt=self._config_opt, save_on_exit=self._save_on_exit, filename=self._filename)

    def mark_dirty(self):
        if False:
            print('Hello World!')
        'Mark this saveable as dirty (having changes).'
        log.save.debug('Marking {} as dirty.'.format(self._name))
        self._dirty = True

    def save(self, is_exit=False, explicit=False, silent=False, force=False):
        if False:
            while True:
                i = 10
        "Save this saveable.\n\n        Args:\n            is_exit: Whether we're currently exiting qutebrowser.\n            explicit: Whether the user explicitly requested this save.\n            silent: Don't write information to log.\n            force: Force saving, no matter what.\n        "
        if self._config_opt is not None and (not config.instance.get(self._config_opt)) and (not explicit) and (not force):
            if not silent:
                log.save.debug('Not saving {name} because autosaving has been disabled by {cfg[0]} -> {cfg[1]}.'.format(name=self._name, cfg=self._config_opt))
            return
        do_save = self._dirty or (self._save_on_exit and is_exit) or force
        if not silent:
            log.save.debug('Save of {} requested - dirty {}, save_on_exit {}, is_exit {}, force {} -> {}'.format(self._name, self._dirty, self._save_on_exit, is_exit, force, do_save))
        if do_save:
            self._save_handler()
            self._dirty = False

class SaveManager(QObject):
    """Responsible to save 'saveables' periodically and on exit.

    Attributes:
        saveables: A dict mapping names to Saveable instances.
        _save_timer: The Timer used to periodically auto-save things.
    """

    def __init__(self, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        self.saveables: MutableMapping[str, Saveable] = collections.OrderedDict()
        self._save_timer = usertypes.Timer(self, name='save-timer')
        self._save_timer.timeout.connect(self.autosave)
        self._set_autosave_interval()
        config.instance.changed.connect(self._set_autosave_interval)

    def __repr__(self):
        if False:
            return 10
        return utils.get_repr(self, saveables=self.saveables)

    @config.change_filter('auto_save.interval')
    def _set_autosave_interval(self):
        if False:
            return 10
        'Set the auto-save interval.'
        interval = config.val.auto_save.interval
        if interval == 0:
            self._save_timer.stop()
        else:
            self._save_timer.setInterval(interval)
            self._save_timer.start()

    def add_saveable(self, name, save, changed=None, config_opt=None, filename=None, dirty=False):
        if False:
            while True:
                i = 10
        "Add a new saveable.\n\n        Args:\n            name: The name to use.\n            save: The function to call to save this saveable.\n            changed: The signal emitted when this saveable changed.\n            config_opt: An option deciding whether to auto-save or not.\n            filename: The filename of the underlying file, so we can force\n                      saving if it doesn't exist.\n            dirty: Whether the saveable is already dirty.\n        "
        if name in self.saveables:
            raise ValueError('Saveable {} already registered!'.format(name))
        saveable = Saveable(name, save, changed, config_opt, filename)
        self.saveables[name] = saveable
        if dirty:
            saveable.mark_dirty()
            QTimer.singleShot(0, saveable.save)

    def save(self, name, is_exit=False, explicit=False, silent=False, force=False):
        if False:
            print('Hello World!')
        "Save a saveable by name.\n\n        Args:\n            name: The name of the saveable to save.\n            is_exit: Whether we're currently exiting qutebrowser.\n            explicit: Whether this save operation was triggered explicitly.\n            silent: Don't write information to log. Used to reduce log spam\n                    when autosaving.\n            force: Force saving, no matter what.\n        "
        self.saveables[name].save(is_exit=is_exit, explicit=explicit, silent=silent, force=force)

    def save_all(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Save all saveables.'
        for saveable in self.saveables:
            self.save(saveable, *args, **kwargs)

    @pyqtSlot()
    def autosave(self):
        if False:
            i = 10
            return i + 15
        'Slot used when the configs are auto-saved.'
        for (key, saveable) in self.saveables.items():
            try:
                saveable.save(silent=True)
            except OSError as e:
                message.error('Failed to auto-save {}: {}'.format(key, e))

    @cmdutils.register(instance='save-manager', name='save', star_args_optional=True)
    def save_command(self, *what):
        if False:
            i = 10
            return i + 15
        'Save configs and state.\n\n        Args:\n            *what: What to save (`config`/`key-config`/`cookies`/...).\n                   If not given, everything is saved.\n        '
        if what:
            explicit = True
        else:
            what = tuple(self.saveables)
            explicit = False
        for key in what:
            if key not in self.saveables:
                message.error('{} is nothing which can be saved'.format(key))
            else:
                try:
                    self.save(key, explicit=explicit, force=True)
                except OSError as e:
                    message.error('Could not save {}: {}'.format(key, e))
        log.save.debug(':save saved {}'.format(', '.join(what)))

    @pyqtSlot()
    def shutdown(self):
        if False:
            return 10
        'Save all saveables when shutting down.'
        for key in self.saveables:
            try:
                self.save(key, is_exit=True)
            except OSError as e:
                error.handle_fatal_exc(e, 'Error while saving!', pre_text='Error while saving {}'.format(key), no_err_windows=objects.args.no_err_windows)