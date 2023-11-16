"""
Set initial window size to 900/640px without use of
any third party GUI framework. On Linux/Mac you can set
window size by calling WindowInfo.SetAsChild. On Windows
you can accomplish this by calling Windows native functions
using the ctypes module.
"""
from cefpython3 import cefpython as cef
import ctypes
import platform

def main():
    if False:
        while True:
            i = 10
    cef.Initialize()
    window_info = cef.WindowInfo()
    parent_handle = 0
    window_info.SetAsChild(parent_handle, [0, 0, 900, 640])
    browser = cef.CreateBrowserSync(url='https://www.google.com/', window_info=window_info, window_title='Window size')
    if platform.system() == 'Windows':
        window_handle = browser.GetOuterWindowHandle()
        insert_after_handle = 0
        SWP_NOMOVE = 2
        ctypes.windll.user32.SetWindowPos(window_handle, insert_after_handle, 0, 0, 900, 640, SWP_NOMOVE)
    cef.MessageLoop()
    del browser
    cef.Shutdown()
if __name__ == '__main__':
    main()