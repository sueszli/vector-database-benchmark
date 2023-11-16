import os
import sys
import pythoncom
import win32com
import win32com.client.makepy
import win32com.test
genList = [('msword8', '{00020905-0000-0000-C000-000000000046}', 1033, 8, 0)]
genDir = 'Generated4Test'

def GetGenPath():
    if False:
        for i in range(10):
            print('nop')
    import win32api
    return os.path.join(win32api.GetFullPathName(win32com.test.__path__[0]), genDir)

def GenerateFromRegistered(fname, *loadArgs):
    if False:
        for i in range(10):
            print('nop')
    genPath = GetGenPath()
    try:
        os.stat(genPath)
    except OSError:
        os.mkdir(genPath)
    open(os.path.join(genPath, '__init__.py'), 'w').close()
    print(fname, ': generating -', end=' ')
    f = open(os.path.join(genPath, fname + '.py'), 'w')
    win32com.client.makepy.GenerateFromTypeLibSpec(loadArgs, f, bQuiet=1, bGUIProgress=1)
    f.close()
    print('compiling -', end=' ')
    fullModName = f'win32com.test.{genDir}.{fname}'
    exec('import ' + fullModName)
    sys.modules[fname] = sys.modules[fullModName]
    print('done')

def GenerateAll():
    if False:
        print('Hello World!')
    for args in genList:
        try:
            GenerateFromRegistered(*args)
        except KeyboardInterrupt:
            print('** Interrupted ***')
            break
        except pythoncom.com_error:
            print('** Could not generate test code for ', args[0])

def CleanAll():
    if False:
        print('Hello World!')
    print('Cleaning generated test scripts...')
    try:
        1 / 0
    except:
        pass
    genPath = GetGenPath()
    for args in genList:
        try:
            name = args[0] + '.py'
            os.unlink(os.path.join(genPath, name))
        except OSError as details:
            if isinstance(details, tuple) and details[0] != 2:
                print('Could not deleted generated', name, details)
        try:
            name = args[0] + '.pyc'
            os.unlink(os.path.join(genPath, name))
        except OSError as details:
            if isinstance(details, tuple) and details[0] != 2:
                print('Could not deleted generated', name, details)
        try:
            os.unlink(os.path.join(genPath, '__init__.py'))
        except:
            pass
        try:
            os.unlink(os.path.join(genPath, '__init__.pyc'))
        except:
            pass
    try:
        os.rmdir(genPath)
    except OSError as details:
        print('Could not delete test directory -', details)
if __name__ == '__main__':
    GenerateAll()
    CleanAll()