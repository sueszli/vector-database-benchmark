"""Some automation of Windows Media player"""
from __future__ import unicode_literals
from __future__ import print_function
import time
import sys
try:
    from pywinauto import application
except ImportError:
    import os.path
    pywinauto_path = os.path.abspath(__file__)
    pywinauto_path = os.path.split(os.path.split(pywinauto_path)[0])[0]
    sys.path.append(pywinauto_path)
    from pywinauto import application

def windows_media():
    if False:
        while True:
            i = 10
    app = application.Application()
    try:
        app.start('C:\\Program Files\\Windows Media Player\\wmplayer.exe')
    except application.ProcessNotFoundError:
        print('You must first start Windows Media Player before running this script')
        sys.exit()
    app.WindowsMediaPlayer.menu_select('View->GoTo->Library')
    app.WindowsMediaPlayer.menu_select('View->Choose Columns')
    print('Is it checked already:', app.ChooseColumsn.ListView.is_checked(1))
    app.ChooseColumns.ListView.check(1)
    time.sleep(0.5)
    print('Shold be checked now:', app.ChooseColumsn.ListView.is_checked(1))
    app.ChooseColumns.ListView.uncheck(1)
    time.sleep(0.5)
    print('Should not be checked now:', app.ChooseColumsn.ListView.is_checked(1))
    app.ChooseColumns.ListView.check(1)
    time.sleep(0.5)
    app.ChooseColumsn.Cancel.click()
    app.WindowsMediaPlayer.menu_select('File->Exit')

def main():
    if False:
        print('Hello World!')
    start = time.time()
    windows_media()
    print('Total time taken:', time.time() - start)
if __name__ == '__main__':
    main()