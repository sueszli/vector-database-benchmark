""" This is our runner for the inline copy of scons.

It dispatches based on the Python version it is running in, with 2.6 using a
very old version. Once scons stops supporting Python2.7 as well, we might have
to add another one.

"""
if __name__ == '__main__':
    import hashlib
    import os
    import sys
    if sys.version_info >= (3, 0) and sys.version_info < (3, 5):
        sys.exit('Error, scons must not be run with Python3 older than 3.5.')
    if sys.version_info < (2, 7):
        scons_version = 'scons-2.3.2'
    elif os.name == 'nt' and sys.version_info >= (3, 5):
        scons_version = 'scons-4.3.0'
    else:
        scons_version = 'scons-3.1.2'
    sys.path.insert(0, os.path.abspath(os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'lib', scons_version))))
    os.environ['SCONS_LIB_DIR'] = sys.path[0]
    try:
        hashlib.md5()
    except ValueError:
        _md5 = hashlib.md5

        def md5(value=b''):
            if False:
                print('Hello World!')
            return _md5(value, usedforsecurity=False)
        hashlib.md5 = md5
    import SCons.Script
    SCons.Script.main()