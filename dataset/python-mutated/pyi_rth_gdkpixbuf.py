def _pyi_rthook():
    if False:
        for i in range(10):
            print('nop')
    import atexit
    import os
    import sys
    import tempfile
    pixbuf_file = os.path.join(sys._MEIPASS, 'lib', 'gdk-pixbuf', 'loaders.cache')
    if os.path.exists(pixbuf_file) and sys.platform != 'win32':
        with open(pixbuf_file, 'rb') as fp:
            contents = fp.read()
        (fd, pixbuf_file) = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as fp:
            libpath = os.path.join(sys._MEIPASS, 'lib').encode('utf-8')
            fp.write(contents.replace(b'@executable_path/lib', libpath))
        try:
            atexit.register(os.unlink, pixbuf_file)
        except OSError:
            pass
    os.environ['GDK_PIXBUF_MODULE_FILE'] = pixbuf_file
_pyi_rthook()
del _pyi_rthook