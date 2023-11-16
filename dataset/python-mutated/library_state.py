"""The core state needed to make use of bzr is managed here."""
from __future__ import absolute_import
__all__ = ['BzrLibraryState']
import bzrlib
from bzrlib.lazy_import import lazy_import
lazy_import(globals(), '\nfrom bzrlib import (\n    cleanup,\n    config,\n    osutils,\n    symbol_versioning,\n    trace,\n    ui,\n    )\n')

class BzrLibraryState(object):
    """The state about how bzrlib has been configured.

    This is the core state needed to make use of bzr. The current instance is
    currently always exposed as bzrlib.global_state, but we desired to move
    to a point where no global state is needed at all.

    :ivar saved_state: The bzrlib.global_state at the time __enter__ was
        called.
    :ivar cleanups: An ObjectWithCleanups which can be used for cleanups that
        should occur when the use of bzrlib is completed. This is initialised
        in __enter__ and executed in __exit__.
    """

    def __init__(self, ui, trace):
        if False:
            for i in range(10):
                print('nop')
        'Create library start for normal use of bzrlib.\n\n        Most applications that embed bzrlib, including bzr itself, should just\n        call bzrlib.initialize(), but it is possible to use the state class\n        directly. The initialize() function provides sensible defaults for a\n        CLI program, such as a text UI factory.\n\n        More options may be added in future so callers should use named\n        arguments.\n\n        BzrLibraryState implements the Python 2.5 Context Manager protocol\n        PEP343, and can be used with the with statement. Upon __enter__ the\n        global variables in use by bzr are set, and they are cleared on\n        __exit__.\n\n        :param ui: A bzrlib.ui.ui_factory to use.\n        :param trace: A bzrlib.trace.Config context manager to use, perhaps\n            bzrlib.trace.DefaultConfig.\n        '
        self._ui = ui
        self._trace = trace
        self.cmdline_overrides = config.CommandLineStore()
        self.config_stores = {}
        self.started = False

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.started:
            self._start()
        return self

    def _start(self):
        if False:
            return 10
        'Do all initialization.'
        self.cleanups = cleanup.ObjectWithCleanups()
        if bzrlib.version_info[3] == 'final':
            self.cleanups.add_cleanup(symbol_versioning.suppress_deprecation_warnings(override=True))
        self._trace.__enter__()
        self._orig_ui = bzrlib.ui.ui_factory
        bzrlib.ui.ui_factory = self._ui
        self._ui.__enter__()
        self.saved_state = bzrlib.global_state
        bzrlib.global_state = self
        self.started = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            return 10
        if exc_type is None:
            for (k, store) in self.config_stores.iteritems():
                store.save_changes()
        self.cleanups.cleanup_now()
        trace._flush_stdout_stderr()
        trace._flush_trace()
        osutils.report_extension_load_failures()
        self._ui.__exit__(None, None, None)
        self._trace.__exit__(None, None, None)
        ui.ui_factory = self._orig_ui
        bzrlib.global_state = self.saved_state
        return False