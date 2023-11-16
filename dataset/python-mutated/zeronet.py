import os
import sys

def main():
    if False:
        i = 10
        return i + 15
    if sys.version_info.major < 3:
        print('Error: Python 3.x is required')
        sys.exit(0)
    if '--silent' not in sys.argv:
        print('- Starting ZeroNet...')
    main = None
    try:
        import main
        main.start()
    except Exception as err:
        import traceback
        try:
            import logging
            logging.exception('Unhandled exception: %s' % err)
        except Exception as log_err:
            print('Failed to log error:', log_err)
            traceback.print_exc()
        from Config import config
        error_log_path = config.log_dir + '/error.log'
        traceback.print_exc(file=open(error_log_path, 'w'))
        print('---')
        print('Please report it: https://github.com/HelloZeroNet/ZeroNet/issues/new?assignees=&labels=&template=bug-report.md')
        if sys.platform.startswith('win') and 'python.exe' not in sys.executable:
            displayErrorMessage(err, error_log_path)
    if main and (main.update_after_shutdown or main.restart_after_shutdown):
        if main.update_after_shutdown:
            print('Shutting down...')
            prepareShutdown()
            import update
            print('Updating...')
            update.update()
            if main.restart_after_shutdown:
                print('Restarting...')
                restart()
        else:
            print('Shutting down...')
            prepareShutdown()
            print('Restarting...')
            restart()

def displayErrorMessage(err, error_log_path):
    if False:
        return 10
    import ctypes
    import urllib.parse
    import subprocess
    MB_YESNOCANCEL = 3
    MB_ICONEXCLAIMATION = 48
    ID_YES = 6
    ID_NO = 7
    ID_CANCEL = 2
    err_message = '%s: %s' % (type(err).__name__, err)
    err_title = 'Unhandled exception: %s\nReport error?' % err_message
    res = ctypes.windll.user32.MessageBoxW(0, err_title, 'ZeroNet error', MB_YESNOCANCEL | MB_ICONEXCLAIMATION)
    if res == ID_YES:
        import webbrowser
        report_url = 'https://github.com/HelloZeroNet/ZeroNet/issues/new?assignees=&labels=&template=bug-report.md&title=%s'
        webbrowser.open(report_url % urllib.parse.quote('Unhandled exception: %s' % err_message))
    if res in [ID_YES, ID_NO]:
        subprocess.Popen(['notepad.exe', error_log_path])

def prepareShutdown():
    if False:
        i = 10
        return i + 15
    import atexit
    atexit._run_exitfuncs()
    if 'main' in sys.modules:
        logger = sys.modules['main'].logging.getLogger()
        for handler in logger.handlers[:]:
            handler.flush()
            handler.close()
            logger.removeHandler(handler)
    import time
    time.sleep(1)

def restart():
    if False:
        return 10
    args = sys.argv[:]
    sys.executable = sys.executable.replace('.pkg', '')
    if not getattr(sys, 'frozen', False):
        args.insert(0, sys.executable)
    if '--open_browser' in args:
        del args[args.index('--open_browser') + 1]
        del args[args.index('--open_browser')]
    if getattr(sys, 'frozen', False):
        pos_first_arg = 1
    else:
        pos_first_arg = 2
    args.insert(pos_first_arg, '--open_browser')
    args.insert(pos_first_arg + 1, 'False')
    if sys.platform == 'win32':
        args = ['"%s"' % arg for arg in args]
    try:
        print('Executing %s %s' % (sys.executable, args))
        os.execv(sys.executable, args)
    except Exception as err:
        print('Execv error: %s' % err)
    print('Bye.')

def start():
    if False:
        return 10
    app_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(app_dir)
    sys.path.insert(0, os.path.join(app_dir, 'src/lib'))
    sys.path.insert(0, os.path.join(app_dir, 'src'))
    if '--update' in sys.argv:
        sys.argv.remove('--update')
        print('Updating...')
        import update
        update.update()
    else:
        main()
if __name__ == '__main__':
    start()