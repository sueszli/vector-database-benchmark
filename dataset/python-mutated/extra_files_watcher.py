from qt.core import QObject, QTimer
from time import monotonic
from typing import NamedTuple, Tuple
from calibre.db.constants import DATA_FILE_PATTERN

class ExtraFile(NamedTuple):
    relpath: str
    mtime: float
    size: int

class ExtraFiles(NamedTuple):
    last_changed_at: float
    files: Tuple[ExtraFile, ...]

class ExtraFilesWatcher(QObject):
    WATCH_FOR = 300
    TICK_INTERVAL = 1

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self.watched_book_ids = {}
        self.timer = QTimer(self)
        self.timer.setInterval(int(self.TICK_INTERVAL * 1000))
        self.timer.timeout.connect(self.check_registered_books)

    def clear(self):
        if False:
            i = 10
            return i + 15
        self.watched_book_ids.clear()
        self.timer.stop()

    def watch_book(self, book_id):
        if False:
            return 10
        if book_id not in self.watched_book_ids:
            try:
                self.watched_book_ids[book_id] = ExtraFiles(monotonic(), self.get_extra_files(book_id))
            except Exception:
                import traceback
                traceback.print_exc()
                return
        self.timer.start()

    @property
    def gui(self):
        if False:
            for i in range(10):
                print('nop')
        ans = self.parent()
        if hasattr(ans, 'current_db'):
            return ans
        from calibre.gui2.ui import get_gui
        return get_gui()

    def get_extra_files(self, book_id):
        if False:
            while True:
                i = 10
        db = self.gui.current_db.new_api
        return tuple((ExtraFile(ef.relpath, ef.stat_result.st_mtime, ef.stat_result.st_size) for ef in db.list_extra_files(book_id, pattern=DATA_FILE_PATTERN)))

    def check_registered_books(self):
        if False:
            print('Hello World!')
        changed = {}
        remove = set()
        now = monotonic()
        for (book_id, extra_files) in self.watched_book_ids.items():
            try:
                ef = self.get_extra_files(book_id)
            except Exception:
                remove.add(book_id)
                continue
            if ef != extra_files.files:
                changed[book_id] = ef
            elif now - extra_files.last_changed_at > self.WATCH_FOR:
                remove.add(book_id)
        if changed:
            self.refresh_gui(changed)
            for (book_id, files) in changed.items():
                self.watched_book_ids[book_id] = self.watched_book_ids[book_id]._replace(files=files, last_changed_at=now)
        for book_id in remove:
            self.watched_book_ids.pop(book_id, None)
        if not self.watched_book_ids:
            self.timer.stop()

    def refresh_gui(self, book_ids):
        if False:
            i = 10
            return i + 15
        lv = self.gui.library_view
        lv.model().refresh_ids(frozenset(book_ids), current_row=lv.currentIndex().row())