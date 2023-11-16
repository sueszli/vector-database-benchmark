import ntpath
import os
import posixpath

def posix_expanduser(path, worker_environ):
    if False:
        while True:
            i = 10
    'Expand ~ and ~user constructions.  If user or $HOME is unknown,\n    do nothing.'
    path = os.fspath(path)
    tilde = '~'
    if not path.startswith(tilde):
        return path
    sep = posixpath._get_sep(path)
    i = path.find(sep, 1)
    if i < 0:
        i = len(path)
    if i == 1:
        if 'HOME' not in worker_environ:
            try:
                import pwd
            except ImportError:
                return path
            try:
                userhome = pwd.getpwuid(os.getuid()).pw_dir
            except KeyError:
                return path
        else:
            userhome = worker_environ['HOME']
    else:
        try:
            import pwd
        except ImportError:
            return path
        name = path[1:i]
        try:
            pwent = pwd.getpwnam(name)
        except KeyError:
            return path
        userhome = pwent.pw_dir
    root = '/'
    userhome = userhome.rstrip(root)
    return userhome + path[i:] or root

def nt_expanduser(path, worker_environ):
    if False:
        print('Hello World!')
    'Expand ~ and ~user constructs.\n    If user or $HOME is unknown, do nothing.'
    path = os.fspath(path)
    tilde = '~'
    if not path.startswith(tilde):
        return path
    (i, n) = (1, len(path))
    while i < n and path[i] not in ntpath._get_bothseps(path):
        i += 1
    if 'USERPROFILE' in worker_environ:
        userhome = worker_environ['USERPROFILE']
    elif 'HOMEPATH' not in worker_environ:
        return path
    else:
        try:
            drive = worker_environ['HOMEDRIVE']
        except KeyError:
            drive = ''
        userhome = ntpath.join(drive, worker_environ['HOMEPATH'])
    if i != 1:
        target_user = path[1:i]
        current_user = worker_environ.get('USERNAME')
        if target_user != current_user:
            if current_user != ntpath.basename(userhome):
                return path
            userhome = ntpath.join(ntpath.dirname(userhome), target_user)
    return userhome + path[i:]