import sys
import __main__
import regutil
import win32api
import win32ui
demos = [('Open GL Demo', 'import openGLDemo;openGLDemo.test()'), ('Threaded GUI', 'import threadedgui;threadedgui.ThreadedDemo()'), ('Tree View Demo', 'import hiertest;hiertest.demoboth()'), ('3-Way Splitter Window', 'import splittst;splittst.demo()'), ('Custom Toolbars and Tooltips', 'import toolbar;toolbar.test()'), ('Progress Bar', 'import progressbar;progressbar.demo()'), ('Slider Control', 'import sliderdemo;sliderdemo.demo()'), ('Dynamic window creation', 'import createwin;createwin.demo()'), ('Various Dialog demos', 'import dlgtest;dlgtest.demo()'), ('OCX Control Demo', 'from ocx import ocxtest;ocxtest.demo()'), ('OCX Serial Port Demo', 'from ocx import ocxserialtest;\tocxserialtest.test()'), ('IE4 Control Demo', 'from ocx import webbrowser; webbrowser.Demo("http://www.python.org")')]

def demo():
    if False:
        for i in range(10):
            print('nop')
    try:
        import fontdemo
    except ImportError:
        try:
            instPath = regutil.GetRegistryDefaultValue(regutil.BuildDefaultPythonKey() + '\\InstallPath')
        except win32api.error:
            print('The InstallPath can not be located, and the Demos directory is not on the path')
            instPath = '.'
        demosDir = win32ui.FullPath(instPath + '\\Demos')
        for path in sys.path:
            if win32ui.FullPath(path) == demosDir:
                break
        else:
            sys.path.append(demosDir)
        import fontdemo
    import sys
    if '/go' in sys.argv:
        for (name, cmd) in demos:
            try:
                exec(cmd)
            except:
                print(f'Demo of {cmd} failed - {sys.exc_info()[0]}:{sys.exc_info()[1]}')
        return
    import pywin.dialogs.list
    while 1:
        rc = pywin.dialogs.list.SelectFromLists('Select a Demo', demos, ['Demo Title'])
        if rc is None:
            break
        (title, cmd) = demos[rc]
        try:
            exec(cmd)
        except:
            print(f'Demo of {title} failed - {sys.exc_info()[0]}:{sys.exc_info()[1]}')
if __name__ == __main__.__name__:
    import demoutils
    if demoutils.NeedGoodGUI():
        demo()