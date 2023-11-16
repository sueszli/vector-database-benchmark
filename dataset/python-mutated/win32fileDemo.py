import os
import win32api
import win32con
import win32file

def SimpleFileDemo():
    if False:
        i = 10
        return i + 15
    testName = os.path.join(win32api.GetTempPath(), 'win32file_demo_test_file')
    if os.path.exists(testName):
        os.unlink(testName)
    handle = win32file.CreateFile(testName, win32file.GENERIC_WRITE, 0, None, win32con.CREATE_NEW, 0, None)
    test_data = b'Hello\x00there'
    win32file.WriteFile(handle, test_data)
    handle.Close()
    handle = win32file.CreateFile(testName, win32file.GENERIC_READ, 0, None, win32con.OPEN_EXISTING, 0, None)
    (rc, data) = win32file.ReadFile(handle, 1024)
    handle.Close()
    if data == test_data:
        print('Successfully wrote and read a file')
    else:
        raise Exception('Got different data back???')
    os.unlink(testName)
if __name__ == '__main__':
    SimpleFileDemo()