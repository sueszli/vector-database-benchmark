from pathlib import Path
from PyQt6.QtCore import QTimer
from picard import log
TOUCH_FILES_DELAY_SECONDS = 4 * 3600
_touch_timer = QTimer()
_files_to_touch = set()

def register_file(filepath):
    if False:
        while True:
            i = 10
    if _touch_timer.isActive():
        _files_to_touch.add(filepath)

def unregister_file(filepath):
    if False:
        while True:
            i = 10
    if _touch_timer.isActive():
        _files_to_touch.discard(filepath)

def enable_timer():
    if False:
        while True:
            i = 10
    log.debug('Setup timer for touching files every %i seconds', TOUCH_FILES_DELAY_SECONDS)
    _touch_timer.timeout.connect(_touch_files)
    _touch_timer.start(TOUCH_FILES_DELAY_SECONDS * 1000)

def _touch_files():
    if False:
        i = 10
        return i + 15
    log.debug('Touching %i files', len(_files_to_touch))
    for filepath in _files_to_touch.copy():
        path = Path(filepath)
        if path.exists():
            try:
                path.touch()
            except OSError:
                log.error('error touching file `%s`', filepath, exc_info=True)
        else:
            unregister_file(filepath)