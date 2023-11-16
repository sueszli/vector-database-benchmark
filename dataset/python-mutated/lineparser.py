"""Parser for line-based files like histories."""
import os
import os.path
import contextlib
from typing import Sequence
from qutebrowser.qt.core import pyqtSlot, pyqtSignal, QObject
from qutebrowser.utils import log, utils, qtutils
from qutebrowser.config import config

class BaseLineParser(QObject):
    """A LineParser without any real data.

    Attributes:
        _configdir: Directory to read the config from, or None.
        _configfile: The config file path.
        _fname: Filename of the config.
        _binary: Whether to open the file in binary mode.

    Signals:
        changed: Emitted when the history was changed.
    """
    changed = pyqtSignal()

    def __init__(self, configdir, fname, *, binary=False, parent=None):
        if False:
            i = 10
            return i + 15
        'Constructor.\n\n        Args:\n            configdir: Directory to read the config from.\n            fname: Filename of the config file.\n            binary: Whether to open the file in binary mode.\n            _opened: Whether the underlying file is open\n        '
        super().__init__(parent)
        self._configdir = configdir
        self._configfile = os.path.join(self._configdir, fname)
        self._fname = fname
        self._binary = binary
        self._opened = False

    def __repr__(self):
        if False:
            print('Hello World!')
        return utils.get_repr(self, constructor=True, configdir=self._configdir, fname=self._fname, binary=self._binary)

    def _prepare_save(self):
        if False:
            return 10
        'Prepare saving of the file.\n\n        Return:\n            True if the file should be saved, False otherwise.\n        '
        os.makedirs(self._configdir, 493, exist_ok=True)
        return True

    def _after_save(self):
        if False:
            while True:
                i = 10
        'Log a message after saving is done.'
        log.destroy.debug('Saved to {}'.format(self._configfile))

    @contextlib.contextmanager
    def _open(self, mode):
        if False:
            return 10
        "Open self._configfile for reading.\n\n        Args:\n            mode: The mode to use ('a'/'r'/'w')\n\n        Raises:\n            OSError: if the file is already open\n\n        Yields:\n            a file object for the config file\n        "
        assert self._configfile is not None
        if self._opened:
            raise OSError('Refusing to double-open LineParser.')
        self._opened = True
        try:
            if self._binary:
                with open(self._configfile, mode + 'b') as f:
                    yield f
            else:
                with open(self._configfile, mode, encoding='utf-8') as f:
                    yield f
        finally:
            self._opened = False

    def _write(self, fp, data):
        if False:
            for i in range(10):
                print('nop')
        'Write the data to a file.\n\n        Args:\n            fp: A file object to write the data to.\n            data: The data to write.\n        '
        if not data:
            return
        if self._binary:
            fp.write(b'\n'.join(data))
            fp.write(b'\n')
        else:
            fp.write('\n'.join(data))
            fp.write('\n')

    def save(self):
        if False:
            for i in range(10):
                print('nop')
        'Save the history to disk.'
        raise NotImplementedError

    def clear(self):
        if False:
            print('Hello World!')
        'Clear the contents of the file.'
        raise NotImplementedError

class LineParser(BaseLineParser):
    """Parser for configuration files which are simply line-based.

    Attributes:
        data: A list of lines.
    """

    def __init__(self, configdir, fname, *, binary=False, parent=None):
        if False:
            while True:
                i = 10
        'Constructor.\n\n        Args:\n            configdir: Directory to read the config from.\n            fname: Filename of the config file.\n            binary: Whether to open the file in binary mode.\n        '
        super().__init__(configdir, fname, binary=binary, parent=parent)
        if not os.path.isfile(self._configfile):
            self.data: Sequence[str] = []
        else:
            log.init.debug('Reading {}'.format(self._configfile))
            self._read()

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return iter(self.data)

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        return self.data[key]

    def _read(self):
        if False:
            return 10
        'Read the data from self._configfile.'
        with self._open('r') as f:
            if self._binary:
                self.data = [line.rstrip(b'\n') for line in f]
            else:
                self.data = [line.rstrip('\n') for line in f]

    def save(self):
        if False:
            print('Hello World!')
        'Save the config file.'
        if self._opened:
            raise OSError('Refusing to double-open LineParser.')
        do_save = self._prepare_save()
        if not do_save:
            return
        self._opened = True
        try:
            assert self._configfile is not None
            with qtutils.savefile_open(self._configfile, self._binary) as f:
                self._write(f, self.data)
        finally:
            self._opened = False
        self._after_save()

    def clear(self):
        if False:
            while True:
                i = 10
        self.data = []
        self.save()

class LimitLineParser(LineParser):
    """A LineParser with a limited count of lines.

    Attributes:
        _limit: The config option used to limit the maximum number of lines.
    """

    def __init__(self, configdir, fname, *, limit, binary=False, parent=None):
        if False:
            return 10
        'Constructor.\n\n        Args:\n            configdir: Directory to read the config from, or None.\n            fname: Filename of the config file.\n            limit: Config option which contains a limit.\n            binary: Whether to open the file in binary mode.\n        '
        super().__init__(configdir, fname, binary=binary, parent=parent)
        self._limit = limit
        if limit is not None and configdir is not None:
            config.instance.changed.connect(self._cleanup_file)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return utils.get_repr(self, constructor=True, configdir=self._configdir, fname=self._fname, limit=self._limit, binary=self._binary)

    @pyqtSlot(str)
    def _cleanup_file(self, option):
        if False:
            while True:
                i = 10
        'Delete the file if the limit was changed to 0.'
        assert self._configfile is not None
        if option != self._limit:
            return
        value = config.instance.get(option)
        if value == 0:
            if os.path.exists(self._configfile):
                os.remove(self._configfile)

    def save(self):
        if False:
            return 10
        'Save the config file.'
        limit = config.instance.get(self._limit)
        if limit == 0:
            return
        do_save = self._prepare_save()
        if not do_save:
            return
        assert self._configfile is not None
        with qtutils.savefile_open(self._configfile, self._binary) as f:
            self._write(f, self.data[-limit:])
        self._after_save()