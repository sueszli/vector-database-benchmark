import os
import time
from visidata import vd, BaseSheet, Sheet, asyncthread, Path, ScopedSetattr

@BaseSheet.api
@asyncthread
def reload_every(sheet, seconds: int):
    if False:
        for i in range(10):
            print('nop')
    while True:
        sheet.reload()
        time.sleep(seconds)

@BaseSheet.api
@asyncthread
def reload_modified(sheet):
    if False:
        while True:
            i = 10
    'Spawn thread to call sheet.reload_rows when sheet.source mtime has changed.'
    p = sheet.source
    assert isinstance(p, Path)
    assert not p.is_url()
    mtime = os.stat(p).st_mtime
    while True:
        time.sleep(1)
        t = os.stat(p).st_mtime
        if t != mtime:
            mtime = t
            sheet.reload_rows()

@Sheet.api
@asyncthread
def reload_rows(self):
    if False:
        print('Hello World!')
    'Reload rows from ``self.source``, keeping current columns intact.  Async.'
    with ScopedSetattr(self, 'loading', True), ScopedSetattr(self, 'checkCursor', lambda : True), ScopedSetattr(self, 'cursorRowIndex', self.cursorRowIndex):
        self.beforeLoad()
        try:
            self.loader()
            vd.status('finished loading rows')
        finally:
            self.afterLoad()
BaseSheet.addCommand('', 'reload-every', 'sheet.reload_every(input("reload interval (sec): ", value=1))', 'schedule sheet reload every N seconds')
BaseSheet.addCommand('', 'reload-modified', 'sheet.reload_modified()', 'reload sheet when source file modified (tail-like behavior)')
BaseSheet.addCommand('z^R', 'reload-rows', 'preloadHook(); reload_rows(); status("reloaded")', 'Reload current sheet')
vd.addMenuItems('\n    File > Reload > rows only > reload-rows\n    File > Reload > every N seconds > reload-every\n    File > Reload > when source modified > reload-modified\n')