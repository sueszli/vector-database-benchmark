"""Managers for bookmarks and quickmarks.

Note we violate our general QUrl rule by storing url strings in the marks
OrderedDict. This is because we read them from a file at start and write them
to a file on shutdown, so it makes sense to keep them as strings here.
"""
import os
import os.path
import html
import functools
import collections
from typing import MutableMapping
from qutebrowser.qt.core import pyqtSignal, QUrl, QObject
from qutebrowser.utils import message, usertypes, qtutils, urlutils, standarddir, objreg, log
from qutebrowser.api import cmdutils
from qutebrowser.misc import lineparser

class Error(Exception):
    """Base class for all errors in this module."""

class InvalidUrlError(Error):
    """Exception emitted when a URL is invalid."""

class DoesNotExistError(Error):
    """Exception emitted when a given URL does not exist."""

class AlreadyExistsError(Error):
    """Exception emitted when a given URL does already exist."""

class UrlMarkManager(QObject):
    """Base class for BookmarkManager and QuickmarkManager.

    Attributes:
        marks: An OrderedDict of all quickmarks/bookmarks.
        _lineparser: The LineParser used for the marks

    Signals:
        changed: Emitted when anything changed.
    """
    changed = pyqtSignal()
    _lineparser: lineparser.LineParser

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        'Initialize and read quickmarks.'
        super().__init__(parent)
        self.marks: MutableMapping[str, str] = collections.OrderedDict()
        self._init_lineparser()
        for line in self._lineparser:
            if not line.strip() or line.startswith('#'):
                continue
            self._parse_line(line)
        self._init_savemanager(objreg.get('save-manager'))

    def _init_lineparser(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def _parse_line(self, line):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def _init_savemanager(self, _save_manager):
        if False:
            return 10
        raise NotImplementedError

    def save(self):
        if False:
            while True:
                i = 10
        'Save the marks to disk.'
        self._lineparser.data = [' '.join(tpl) for tpl in self.marks.items()]
        self._lineparser.save()

    def delete(self, key):
        if False:
            for i in range(10):
                print('nop')
        'Delete a quickmark/bookmark.\n\n        Args:\n            key: The key to delete (name for quickmarks, URL for bookmarks.)\n        '
        del self.marks[key]
        self.changed.emit()

    def clear(self):
        if False:
            i = 10
            return i + 15
        'Delete all marks.'
        self.marks.clear()
        self.changed.emit()

class QuickmarkManager(UrlMarkManager):
    """Manager for quickmarks.

    The primary key for quickmarks is their *name*, this means:

        - self.marks maps names to URLs.
        - changed gets emitted with the name as first argument and the URL as
          second argument.
    """

    def _init_lineparser(self):
        if False:
            while True:
                i = 10
        self._lineparser = lineparser.LineParser(standarddir.config(), 'quickmarks', parent=self)

    def _init_savemanager(self, save_manager):
        if False:
            return 10
        filename = os.path.join(standarddir.config(), 'quickmarks')
        save_manager.add_saveable('quickmark-manager', self.save, self.changed, filename=filename)

    def _parse_line(self, line):
        if False:
            while True:
                i = 10
        try:
            (key, url) = line.rsplit(maxsplit=1)
        except ValueError:
            message.error("Invalid quickmark '{}'".format(line))
        else:
            self.marks[key] = url

    def prompt_save(self, url):
        if False:
            while True:
                i = 10
        'Prompt for a new quickmark name to be added and add it.\n\n        Args:\n            url: The quickmark url as a QUrl.\n        '
        if not url.isValid():
            urlutils.invalid_url_error(url, 'save quickmark')
            return
        urlstr = url.toString(QUrl.UrlFormattingOption.RemovePassword | QUrl.ComponentFormattingOption.FullyEncoded)
        message.ask_async('Add quickmark:', usertypes.PromptMode.text, functools.partial(self.quickmark_add, urlstr), text='Please enter a quickmark name for<br/><b>{}</b>'.format(html.escape(url.toDisplayString())), url=urlstr)

    @cmdutils.register(instance='quickmark-manager')
    def quickmark_add(self, url, name):
        if False:
            while True:
                i = 10
        'Add a new quickmark.\n\n        You can view all saved quickmarks on the\n        link:qute://bookmarks[bookmarks page].\n\n        Args:\n            url: The url to add as quickmark.\n            name: The name for the new quickmark.\n        '
        if not name:
            message.error("Can't set mark with empty name!")
            return
        if not url:
            message.error("Can't set mark with empty URL!")
            return

        def set_mark():
            if False:
                return 10
            'Really set the quickmark.'
            self.marks[name] = url
            self.changed.emit()
            log.misc.debug('Added quickmark {} for {}'.format(name, url))
        if name in self.marks:
            message.confirm_async(title='Override existing quickmark?', yes_action=set_mark, default=True, url=url)
        else:
            set_mark()

    def get_by_qurl(self, url):
        if False:
            return 10
        'Look up a quickmark by QUrl, returning its name.\n\n        Takes O(n) time, where n is the number of quickmarks.\n        Use a name instead where possible.\n        '
        qtutils.ensure_valid(url)
        urlstr = url.toString(QUrl.UrlFormattingOption.RemovePassword | QUrl.ComponentFormattingOption.FullyEncoded)
        try:
            index = list(self.marks.values()).index(urlstr)
            key = list(self.marks.keys())[index]
        except ValueError:
            raise DoesNotExistError("Quickmark for '{}' not found!".format(urlstr))
        return key

    def get(self, name):
        if False:
            i = 10
            return i + 15
        'Get the URL of the quickmark named name as a QUrl.'
        if name not in self.marks:
            raise DoesNotExistError("Quickmark '{}' does not exist!".format(name))
        urlstr = self.marks[name]
        try:
            url = urlutils.fuzzy_url(urlstr, do_search=False)
        except urlutils.InvalidUrlError as e:
            raise InvalidUrlError('Invalid URL for quickmark {}: {}'.format(name, str(e)))
        return url

class BookmarkManager(UrlMarkManager):
    """Manager for bookmarks.

    The primary key for bookmarks is their *url*, this means:

        - self.marks maps URLs to titles.
        - changed gets emitted with the URL as first argument and the title as
          second argument.
    """

    def _init_lineparser(self):
        if False:
            print('Hello World!')
        bookmarks_directory = os.path.join(standarddir.config(), 'bookmarks')
        os.makedirs(bookmarks_directory, exist_ok=True)
        bookmarks_subdir = os.path.join('bookmarks', 'urls')
        self._lineparser = lineparser.LineParser(standarddir.config(), bookmarks_subdir, parent=self)

    def _init_savemanager(self, save_manager):
        if False:
            while True:
                i = 10
        filename = os.path.join(standarddir.config(), 'bookmarks', 'urls')
        save_manager.add_saveable('bookmark-manager', self.save, self.changed, filename=filename)

    def _parse_line(self, line):
        if False:
            i = 10
            return i + 15
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            self.marks[parts[0]] = parts[1]
        elif len(parts) == 1:
            self.marks[parts[0]] = ''

    def add(self, url, title, *, toggle=False):
        if False:
            print('Hello World!')
        'Add a new bookmark.\n\n        Args:\n            url: The url to add as bookmark.\n            title: The title for the new bookmark.\n            toggle: remove the bookmark instead of raising an error if it\n                    already exists.\n\n        Return:\n            True if the bookmark was added, and False if it was\n            removed (only possible if toggle is True).\n        '
        if not url.isValid():
            errstr = urlutils.get_errstring(url)
            raise InvalidUrlError(errstr)
        urlstr = url.toString(QUrl.UrlFormattingOption.RemovePassword | QUrl.ComponentFormattingOption.FullyEncoded)
        if urlstr in self.marks:
            if toggle:
                self.delete(urlstr)
                return False
            else:
                raise AlreadyExistsError('Bookmark already exists!')
        else:
            self.marks[urlstr] = title
            self.changed.emit()
            return True