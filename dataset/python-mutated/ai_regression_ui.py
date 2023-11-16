from tkinter import *
import sys
import os
import platform
import run
import subprocess
import result_checker as rc
INFO = 0
WARN = 1
ERROR = 2

def log(sev, msg):
    if False:
        i = 10
        return i + 15
    '\n    This function is used to log info, warnings and errors.\n    '
    logEntry = ''
    if sev == 0:
        logEntry = logEntry + '[INFO]: '
    elif sev == 1:
        logEntry = logEntry + '[WARN]: '
    elif sev == 2:
        logEntry = logEntry + '[ERR] : '
    logEntry = logEntry + str(msg)
    print(logEntry)

class BaseDialog(Toplevel):
    """
    Helper base class for dialogs used in the UI.
    """

    def __init__(self, parent, title=None, buttons=''):
        if False:
            while True:
                i = 10
        '\n        Constructor\n        '
        Toplevel.__init__(self, parent)
        self.transient(parent)
        if title:
            self.title(title)
        self.parent = parent
        self.result = None
        body = Frame(self)
        self.initial_focus = self.body(body)
        body.pack(padx=5, pady=5)
        self.buttonbox(buttons)
        self.grab_set()
        if not self.initial_focus:
            self.initial_focus = self
        self.protocol('WM_DELETE_WINDOW', self.cancel)
        self.geometry('+%d+%d' % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
        self.initial_focus.focus_set()
        self.wait_window(self)

    def body(self, master):
        if False:
            i = 10
            return i + 15
        pass

    def buttonbox(self, buttons):
        if False:
            i = 10
            return i + 15
        box = Frame(self)
        w = Button(box, text='OK', width=40, command=self.ok, default=ACTIVE)
        w.pack(side=LEFT, padx=5, pady=5)
        self.bind('<Return>', self.ok)
        box.pack()

    def ok(self, event=None):
        if False:
            for i in range(10):
                print('nop')
        if not self.validate():
            self.initial_focus.focus_set()
            return
        self.withdraw()
        self.update_idletasks()
        self.apply()
        self.cancel()

    def cancel(self, event=None):
        if False:
            i = 10
            return i + 15
        self.parent.focus_set()
        self.destroy()

    def validate(self):
        if False:
            return 10
        return 1

    def apply(self):
        if False:
            while True:
                i = 10
        pass

class VersionDialog(BaseDialog):
    """
    This class is used to create the info dialog.
    """

    def body(self, master):
        if False:
            print('Hello World!')
        version = 'Asset importer lib version unknown'
        exe = run.getEnvVar('assimp_path')
        if len(exe) != 0:
            command = [exe, 'version']
            log(INFO, 'command = ' + str(command))
            stdout = subprocess.check_output(command)
            for line in stdout.splitlines():
                pos = str(line).find('Version')
                if -1 != pos:
                    version = line
        Label(master, text=version).pack()

    def apply(self):
        if False:
            while True:
                i = 10
        pass

class SetupDialog(BaseDialog):
    """
    This class is used to create the setup dialog.
    """

    def body(self, master):
        if False:
            i = 10
            return i + 15
        Label(master, justify=LEFT, text='Assimp: ').grid(row=0, column=0)
        Label(master, justify=LEFT, text=run.getEnvVar('assimp_path')).grid(row=0, column=1)
        Label(master, text='New executable:').grid(row=1)
        self.e1 = Entry(master)
        self.e1.grid(row=1, column=1)
        return self.e1

    def apply(self):
        if False:
            print('Hello World!')
        exe = str(self.e1.get())
        if len(exe) == 0:
            return 0
        if os.path.isfile(exe):
            log(INFO, 'Set executable at ' + exe)
            self.assimp_bin_path = exe
            run.setEnvVar('assimp_path', self.assimp_bin_path)
        else:
            log(ERROR, 'Executable not found at ' + exe)
        return 0

class RegDialog(object):
    """
    This class is used to create a simplified user interface for running the regression test suite.
    """

    def __init__(self, bin_path):
        if False:
            print('Hello World!')
        '\n        Constructs the dialog, you can define which executable shal be used.\n        @param  bin_path    [in] Path to assimp binary.\n        '
        run.setEnvVar('assimp_path', bin_path)
        self.b_run_ = None
        self.b_update_ = None
        self.b_res_checker_ = None
        self.b_quit_ = None
        if platform.system() == 'Windows':
            self.editor = 'notepad'
        elif platform.system() == 'Linux':
            self.editor = 'vim'
        self.root = None
        self.width = 40

    def run_reg(self):
        if False:
            while True:
                i = 10
        log(INFO, 'Starting regression test suite.')
        run.run_test()
        rc.run()
        self.b_update_.config(state=ACTIVE)
        return 0

    def reg_update(self):
        if False:
            i = 10
            return i + 15
        assimp_exe = run.getEnvVar('assimp_path')
        if len(assimp_exe) == 0:
            return 1
        exe = 'python'
        command = [exe, 'gen_db.py', assimp_exe]
        log(INFO, 'command = ' + str(command))
        stdout = subprocess.call(command)
        log(INFO, stdout)
        return 0

    def shop_diff(self):
        if False:
            return 10
        log(WARN, 'ToDo!')
        return 0

    def open_log(self):
        if False:
            while True:
                i = 10
        command = [self.editor, '../results/run_regression_suite_output.txt']
        log(INFO, 'command = ' + str(command))
        r = subprocess.call(command)
        return 0

    def show_version(self):
        if False:
            print('Hello World!')
        d = VersionDialog(self.root)
        return 0

    def setup(self):
        if False:
            return 10
        d = SetupDialog(self.root)
        return 0

    def quit(self):
        if False:
            for i in range(10):
                print('nop')
        log(INFO, 'quit')
        sys.exit(0)

    def initUi(self):
        if False:
            while True:
                i = 10
        self.root = Tk()
        self.root.title('Assimp-Regression UI')
        self.b_run_ = Button(self.root, text='Run regression ', command=self.run_reg, width=self.width)
        self.b_update_ = Button(self.root, text='Update database', command=self.reg_update, width=self.width)
        self.b_show_diff_ = Button(self.root, text='Show diff', command=self.shop_diff, width=self.width)
        self.b_log_ = Button(self.root, text='Open log', command=self.open_log, width=self.width)
        self.b_setup_ = Button(self.root, text='Setup', command=self.setup, width=self.width)
        self.b_version_ = Button(self.root, text='Show version', command=self.show_version, width=self.width)
        self.b_quit_ = Button(self.root, text='Quit', command=self.quit, width=self.width)
        self.b_run_.grid(row=0, column=0, sticky=W + E)
        self.b_update_.grid(row=1, column=0, sticky=W + E)
        self.b_show_diff_.grid(row=2, column=0, sticky=W + E)
        self.b_log_.grid(row=3, column=0, sticky=W + E)
        self.b_setup_.grid(row=4, column=0, sticky=W + E)
        self.b_version_.grid(row=5, column=0, sticky=W + E)
        self.b_quit_.grid(row=6, column=0, sticky=W + E)
        self.b_show_diff_.config(state=DISABLED)
        self.root.mainloop()

def getDefaultExecutable():
    if False:
        while True:
            i = 10
    assimp_bin_path = ''
    if platform.system() == 'Windows':
        assimp_bin_path = '..\\..\\bin\\debug\\assimpd.exe'
    elif platform.system() == 'Linux':
        assimp_bin_path = '../../bin/assimp'
    return assimp_bin_path
if __name__ == '__main__':
    if len(sys.argv) > 1:
        assimp_bin_path = sys.argv[1]
    else:
        assimp_bin_path = getDefaultExecutable()
    log(INFO, 'Using assimp binary: ' + assimp_bin_path)
    dlg = RegDialog(assimp_bin_path)
    dlg.initUi()