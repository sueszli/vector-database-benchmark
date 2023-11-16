def _samefile(os, src, dst):
    if False:
        return 10
    if hasattr(os.path, 'samefile'):
        try:
            return os.path.samefile(src, dst)
        except OSError:
            return False
    return os.path.normcase(os.path.abspath(src)) == os.path.normcase(os.path.abspath(dst))