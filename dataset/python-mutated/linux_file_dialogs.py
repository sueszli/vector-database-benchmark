import functools
import os
import subprocess
import sys
import time
from threading import Thread
from qt.core import QEventLoop
from calibre import force_unicode
from calibre.constants import DEBUG, filesystem_encoding, preferred_encoding
from calibre.utils.config import dynamic
from polyglot.builtins import reraise, string_or_bytes

def dialog_name(name, title):
    if False:
        i = 10
        return i + 15
    return name or 'dialog_' + title

def get_winid(widget=None):
    if False:
        return 10
    if widget is not None:
        return widget.effectiveWinId()

def detect_desktop_environment():
    if False:
        for i in range(10):
            print('nop')
    de = os.getenv('XDG_CURRENT_DESKTOP')
    if de:
        return de.upper().split(':', 1)[0]
    if os.getenv('KDE_FULL_SESSION') == 'true':
        return 'KDE'
    if os.getenv('GNOME_DESKTOP_SESSION_ID'):
        return 'GNOME'
    ds = os.getenv('DESKTOP_SESSION')
    if ds and ds.upper() in {'GNOME', 'XFCE'}:
        return ds.upper()

def is_executable_present(name):
    if False:
        while True:
            i = 10
    PATH = os.getenv('PATH') or ''
    for path in PATH.split(os.pathsep):
        if os.access(os.path.join(path, name), os.X_OK):
            return True
    return False

def process_path(x):
    if False:
        print('Hello World!')
    if isinstance(x, bytes):
        x = x.decode(filesystem_encoding)
    return os.path.abspath(os.path.expanduser(x))

def ensure_dir(path, default='~'):
    if False:
        print('Hello World!')
    while path and path != '/' and (not os.path.isdir(path)):
        path = os.path.dirname(path)
    if path == '/':
        path = os.path.expanduser(default)
    return path or os.path.expanduser(default)

def get_initial_dir(name, title, default_dir, no_save_dir):
    if False:
        return 10
    if no_save_dir:
        return ensure_dir(process_path(default_dir))
    key = dialog_name(name, title)
    saved = dynamic.get(key)
    if not isinstance(saved, string_or_bytes):
        saved = None
    if saved and os.path.isdir(saved):
        return ensure_dir(process_path(saved))
    return ensure_dir(process_path(default_dir))

def save_initial_dir(name, title, ans, no_save_dir, is_file=False):
    if False:
        while True:
            i = 10
    if ans and (not no_save_dir):
        if is_file:
            ans = os.path.dirname(os.path.abspath(ans))
        key = dialog_name(name, title)
        dynamic.set(key, ans)

def encode_arg(title):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(title, str):
        try:
            title = title.encode(preferred_encoding)
        except UnicodeEncodeError:
            title = title.encode('utf-8')
    return title

def image_extensions():
    if False:
        print('Hello World!')
    from calibre.gui2.dnd import image_extensions
    return image_extensions()

def decode_output(raw):
    if False:
        while True:
            i = 10
    raw = raw or b''
    try:
        return raw.decode(preferred_encoding)
    except UnicodeDecodeError:
        return force_unicode(raw, 'utf-8')

def run(cmd):
    if False:
        while True:
            i = 10
    from calibre.gui2 import sanitize_env_vars
    if DEBUG:
        try:
            print(cmd)
        except Exception:
            pass
    with sanitize_env_vars():
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = p.communicate()
    ret = p.wait()
    return (ret, decode_output(stdout), decode_output(stderr))

def kdialog_supports_desktopfile():
    if False:
        for i in range(10):
            print('nop')
    ans = getattr(kdialog_supports_desktopfile, 'ans', None)
    if ans is None:
        from calibre.gui2 import sanitize_env_vars
        try:
            with sanitize_env_vars():
                raw = subprocess.check_output(['kdialog', '--help'])
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            raw = b'--desktopfile'
        ans = kdialog_supports_desktopfile.ans = b'--desktopfile' in raw
    return ans

def kde_cmd(window, title, *rest):
    if False:
        print('Hello World!')
    ans = ['kdialog', '--title', title]
    if kdialog_supports_desktopfile():
        ans += ['--desktopfile', 'calibre-gui']
    winid = get_winid(window)
    if winid is not None:
        ans += ['--attach', str(int(winid))]
    return ans + list(rest)

def run_kde(cmd):
    if False:
        for i in range(10):
            print('nop')
    (ret, stdout, stderr) = run(cmd)
    if ret == 1:
        return
    if ret != 0:
        raise ValueError(f'KDE file dialog aborted with return code: {ret} and stderr: {stderr}')
    ans = stdout.splitlines()
    return ans

def kdialog_choose_dir(window, name, title, default_dir='~', no_save_dir=False):
    if False:
        for i in range(10):
            print('nop')
    initial_dir = get_initial_dir(name, title, default_dir, no_save_dir)
    ans = run_kde(kde_cmd(window, title, '--getexistingdirectory', initial_dir))
    ans = None if ans is None else ans[0]
    save_initial_dir(name, title, ans, no_save_dir)
    return ans

def kdialog_filters(filters, all_files=True):
    if False:
        i = 10
        return i + 15
    ans = []
    for (name, exts) in filters:
        if not exts or (len(exts) == 1 and exts[0] == '*'):
            ans.append(name + ' (*)')
        else:
            ans.append('{} ({})'.format(name, ' '.join(('*.' + x for x in exts))))
    if all_files:
        ans.append(_('All files') + ' (*)')
    return '\n'.join(ans)

def kdialog_choose_files(window, name, title, filters=[], all_files=True, select_only_single_file=False, default_dir='~', no_save_dir=False):
    if False:
        return 10
    initial_dir = get_initial_dir(name, title, default_dir, no_save_dir)
    args = []
    if not select_only_single_file:
        args += '--multiple --separate-output'.split()
    args += ['--getopenfilename', initial_dir, kdialog_filters(filters, all_files)]
    ans = run_kde(kde_cmd(window, title, *args))
    if not no_save_dir:
        save_initial_dir(name, title, ans[0] if ans else None, False, is_file=True)
    return ans

def kdialog_choose_save_file(window, name, title, filters=[], all_files=True, initial_path=None, initial_filename=None):
    if False:
        return 10
    if initial_path is not None:
        initial_dir = initial_path
    else:
        initial_dir = get_initial_dir(name, title, '~', False)
        if initial_filename:
            initial_dir = os.path.join(initial_dir, initial_filename)
    args = ['--getsavefilename', initial_dir, kdialog_filters(filters, all_files)]
    ans = run_kde(kde_cmd(window, title, *args))
    ans = None if ans is None else ans[0]
    if initial_path is None:
        save_initial_dir(name, title, ans, False, is_file=True)
    return ans

def kdialog_choose_images(window, name, title, select_only_single_file=True, formats=None):
    if False:
        return 10
    return kdialog_choose_files(window, name, title, select_only_single_file=select_only_single_file, all_files=False, filters=[(_('Images'), list(formats or image_extensions()))])

def zenity_cmd(window, title, *rest):
    if False:
        while True:
            i = 10
    ans = ['zenity', '--modal', '--file-selection', '--title=' + title, '--separator=\n']
    winid = get_winid(window)
    if winid is not None:
        ans += ['--attach=%d' % int(winid)]
    return ans + list(rest)

def run_zenity(cmd):
    if False:
        i = 10
        return i + 15
    (ret, stdout, stderr) = run(cmd)
    if ret == 1:
        return
    if ret != 0:
        raise ValueError(f'GTK file dialog aborted with return code: {ret} and stderr: {stderr}')
    ans = stdout.splitlines()
    return ans

def zenity_choose_dir(window, name, title, default_dir='~', no_save_dir=False):
    if False:
        print('Hello World!')
    initial_dir = get_initial_dir(name, title, default_dir, no_save_dir)
    ans = run_zenity(zenity_cmd(window, title, '--directory', '--filename', initial_dir))
    ans = None if ans is None else ans[0]
    save_initial_dir(name, title, ans, no_save_dir)
    return ans

def zenity_filters(filters, all_files=True):
    if False:
        for i in range(10):
            print('nop')
    ans = []
    for (name, exts) in filters:
        if not exts or (len(exts) == 1 and exts[0] == '*'):
            ans.append('--file-filter={} | {}'.format(name, '*'))
        else:
            ans.append('--file-filter={} | {}'.format(name, ' '.join(('*.' + x for x in exts))))
    if all_files:
        ans.append('--file-filter={} | {}'.format(_('All files'), '*'))
    return ans

def zenity_choose_files(window, name, title, filters=[], all_files=True, select_only_single_file=False, default_dir='~', no_save_dir=False):
    if False:
        for i in range(10):
            print('nop')
    initial_dir = get_initial_dir(name, title, default_dir, no_save_dir)
    args = ['--filename=' + os.path.join(initial_dir, '.fgdfg.gdfhjdhf*&^839')]
    args += zenity_filters(filters, all_files)
    if not select_only_single_file:
        args.append('--multiple')
    ans = run_zenity(zenity_cmd(window, title, *args))
    if not no_save_dir:
        save_initial_dir(name, title, ans[0] if ans else None, False, is_file=True)
    return ans

def zenity_choose_save_file(window, name, title, filters=[], all_files=True, initial_path=None, initial_filename=None):
    if False:
        print('Hello World!')
    if initial_path is not None:
        initial_dir = initial_path
    else:
        initial_dir = get_initial_dir(name, title, '~', False)
        initial_dir = os.path.join(initial_dir, initial_filename or _('File name'))
    args = ['--filename=' + initial_dir, '--confirm-overwrite', '--save']
    args += zenity_filters(filters, all_files)
    ans = run_zenity(zenity_cmd(window, title, *args))
    ans = None if ans is None else ans[0]
    if initial_path is None:
        save_initial_dir(name, title, ans, False, is_file=True)
    return ans

def zenity_choose_images(window, name, title, select_only_single_file=True, formats=None):
    if False:
        i = 10
        return i + 15
    return zenity_choose_files(window, name, title, select_only_single_file=select_only_single_file, all_files=False, filters=[(_('Images'), list(formats or image_extensions()))])

def linux_native_dialog(name):
    if False:
        for i in range(10):
            print('nop')
    prefix = check_for_linux_native_dialogs()
    func = globals()[f'{prefix}_choose_{name}']

    @functools.wraps(func)
    def looped(window, *args, **kwargs):
        if False:
            print('Hello World!')
        if hasattr(linux_native_dialog, 'native_failed'):
            import importlib
            m = importlib.import_module('calibre.gui2.qt_file_dialogs')
            qfunc = getattr(m, 'choose_' + name)
            return qfunc(window, *args, **kwargs)
        try:
            if window is None:
                return func(window, *args, **kwargs)
            ret = [None, None]
            loop = QEventLoop(window)

            def r():
                if False:
                    for i in range(10):
                        print('nop')
                try:
                    ret[0] = func(window, *args, **kwargs)
                except:
                    ret[1] = sys.exc_info()
                while not loop.isRunning():
                    time.sleep(0.001)
                loop.quit()
            t = Thread(name='FileDialogHelper', target=r)
            t.daemon = True
            t.start()
            loop.exec(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
            if ret[1] is not None:
                reraise(*ret[1])
            return ret[0]
        except Exception:
            linux_native_dialog.native_failed = True
            import traceback
            traceback.print_exc()
            return looped(window, *args, **kwargs)
    return looped

def check_for_linux_native_dialogs():
    if False:
        print('Hello World!')
    ans = getattr(check_for_linux_native_dialogs, 'ans', None)
    if ans is None:
        de = detect_desktop_environment()
        order = ('zenity', 'kdialog')
        if de in {'GNOME', 'UNITY', 'MATE', 'XFCE'}:
            order = ('zenity',)
        elif de in {'KDE', 'LXDE'}:
            order = ('kdialog',)
        for exe in order:
            if is_executable_present(exe):
                ans = exe
                break
        else:
            ans = False
        check_for_linux_native_dialogs.ans = ans
    return ans
if __name__ == '__main__':
    print(repr(kdialog_choose_files(None, 'testkddcf', 'Testing choose files...', select_only_single_file=False, filters=[('moo', 'epub png'.split()), ('boo', 'docx'.split())], all_files=True)))