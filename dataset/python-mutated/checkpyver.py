"""Check if qutebrowser is run with the correct python version.

This should import and run fine with both python2 and python3.
"""
import sys
try:
    from tkinter import Tk, messagebox
except ImportError:
    try:
        from Tkinter import Tk
        import tkMessageBox as messagebox
    except ImportError:
        Tk = None
        messagebox = None

def check_python_version():
    if False:
        for i in range(10):
            print('nop')
    'Check if correct python version is run.'
    if sys.hexversion < 50855936:
        version_str = '.'.join(map(str, sys.version_info[:3]))
        text = 'At least Python 3.8 is required to run qutebrowser, but ' + "it's running with " + version_str + '.\n'
        show_errors = '--no-err-windows' not in sys.argv
        if Tk and show_errors:
            root = Tk()
            root.withdraw()
            messagebox.showerror('qutebrowser: Fatal error!', text)
        else:
            sys.stderr.write(text)
            sys.stderr.flush()
        sys.exit(1)
if __name__ == '__main__':
    check_python_version()