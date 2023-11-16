"""
Decides if vendor bundles are used or not.
Setup python path accordingly.
"""
from __future__ import absolute_import, print_function
import os.path
import sys
HERE = os.path.dirname(__file__)
TASKS_VENDOR_DIR = os.path.join(HERE, '_vendor')
INVOKE_BUNDLE = os.path.join(TASKS_VENDOR_DIR, 'invoke.zip')
INVOKE_BUNDLE_VERSION = '1.4.0'
DEBUG_SYSPATH = False

class VersionRequirementError(SystemExit):
    pass

def setup_path(invoke_minversion=None):
    if False:
        return 10
    'Setup python search and add ``TASKS_VENDOR_DIR`` (if available).'
    if not os.path.isdir(TASKS_VENDOR_DIR):
        return
    elif os.path.abspath(TASKS_VENDOR_DIR) in sys.path:
        pass
    use_vendor_bundles = os.environ.get('INVOKE_TASKS_USE_VENDOR_BUNDLES', 'no')
    if need_vendor_bundles(invoke_minversion):
        use_vendor_bundles = 'yes'
    if use_vendor_bundles == 'yes':
        syspath_insert(0, os.path.abspath(TASKS_VENDOR_DIR))
        if setup_path_for_bundle(INVOKE_BUNDLE, pos=1):
            import invoke
            bundle_path = os.path.relpath(INVOKE_BUNDLE, os.getcwd())
            print('USING: %s (version: %s)' % (bundle_path, invoke.__version__))
    else:
        syspath_append(os.path.abspath(TASKS_VENDOR_DIR))
        setup_path_for_bundle(INVOKE_BUNDLE, pos=len(sys.path))
    if DEBUG_SYSPATH:
        for (index, p) in enumerate(sys.path):
            print('  %d.  %s' % (index, p))

def require_invoke_minversion(min_version, verbose=False):
    if False:
        print('Hello World!')
    'Ensures that :mod:`invoke` has at the least the :param:`min_version`.\n    Otherwise,\n\n    :param min_version: Minimal acceptable invoke version (as string).\n    :param verbose:     Indicates if invoke.version should be shown.\n    :raises: VersionRequirementError=SystemExit if requirement fails.\n    '
    try:
        import invoke
        invoke_version = invoke.__version__
    except ImportError:
        invoke_version = '__NOT_INSTALLED'
    if invoke_version < min_version:
        message = 'REQUIRE: invoke.version >= %s (but was: %s)' % (min_version, invoke_version)
        message += '\nUSE: pip install invoke>=%s' % min_version
        raise VersionRequirementError(message)
    INVOKE_VERSION = os.environ.get('INVOKE_VERSION', None)
    if verbose and (not INVOKE_VERSION):
        os.environ['INVOKE_VERSION'] = invoke_version
        print('USING: invoke.version=%s' % invoke_version)

def need_vendor_bundles(invoke_minversion=None):
    if False:
        print('Hello World!')
    invoke_minversion = invoke_minversion or '0.0.0'
    need_vendor_answers = []
    need_vendor_answers.append(need_vendor_bundle_invoke(invoke_minversion))
    try:
        import path
        need_bundle = False
    except ImportError:
        need_bundle = True
    need_vendor_answers.append(need_bundle)
    return any(need_vendor_answers)

def need_vendor_bundle_invoke(invoke_minversion='0.0.0'):
    if False:
        i = 10
        return i + 15
    try:
        import invoke
        need_bundle = invoke.__version__ < invoke_minversion
        if need_bundle:
            del sys.modules['invoke']
            del invoke
    except ImportError:
        need_bundle = True
    except Exception:
        need_bundle = True
    return need_bundle

def setup_path_for_bundle(bundle_path, pos=0):
    if False:
        while True:
            i = 10
    if os.path.exists(bundle_path):
        syspath_insert(pos, os.path.abspath(bundle_path))
        return True
    return False

def syspath_insert(pos, path):
    if False:
        while True:
            i = 10
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(pos, path)

def syspath_append(path):
    if False:
        while True:
            i = 10
    if path in sys.path:
        sys.path.remove(path)
    sys.path.append(path)