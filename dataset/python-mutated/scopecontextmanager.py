import logging
from types import TracebackType
from typing import Optional, Type
from opentracing import Scope, ScopeManager, Span
import twisted
from synapse.logging.context import LoggingContext, current_context, nested_logging_context
logger = logging.getLogger(__name__)

class LogContextScopeManager(ScopeManager):
    """
    The LogContextScopeManager tracks the active scope in opentracing
    by using the log contexts which are native to synapse. This is so
    that the basic opentracing api can be used across twisted defereds.

    It would be nice just to use opentracing's ContextVarsScopeManager,
    but currently that doesn't work due to https://twistedmatrix.com/trac/ticket/10301.
    """

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        pass

    @property
    def active(self) -> Optional[Scope]:
        if False:
            while True:
                i = 10
        '\n        Returns the currently active Scope which can be used to access the\n        currently active Scope.span.\n        If there is a non-null Scope, its wrapped Span\n        becomes an implicit parent of any newly-created Span at\n        Tracer.start_active_span() time.\n\n        Return:\n            The Scope that is active, or None if not available.\n        '
        ctx = current_context()
        return ctx.scope

    def activate(self, span: Span, finish_on_close: bool) -> Scope:
        if False:
            return 10
        '\n        Makes a Span active.\n        Args\n            span: the span that should become active.\n            finish_on_close: whether Span should be automatically finished when\n                Scope.close() is called.\n\n        Returns:\n            Scope to control the end of the active period for\n            *span*. It is a programming error to neglect to call\n            Scope.close() on the returned instance.\n        '
        ctx = current_context()
        if not ctx:
            logger.error('Tried to activate scope outside of loggingcontext')
            return Scope(None, span)
        if ctx.scope is not None:
            ctx = nested_logging_context('')
            enter_logcontext = True
        else:
            enter_logcontext = False
        scope = _LogContextScope(self, span, ctx, enter_logcontext, finish_on_close)
        ctx.scope = scope
        if enter_logcontext:
            ctx.__enter__()
        return scope

class _LogContextScope(Scope):
    """
    A custom opentracing scope, associated with a LogContext

      * filters out _DefGen_Return exceptions which arise from calling
        `defer.returnValue` in Twisted code

      * When the scope is closed, the logcontext's active scope is reset to None.
        and - if enter_logcontext was set - the logcontext is finished too.
    """

    def __init__(self, manager: LogContextScopeManager, span: Span, logcontext: LoggingContext, enter_logcontext: bool, finish_on_close: bool):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            manager:\n                the manager that is responsible for this scope.\n            span:\n                the opentracing span which this scope represents the local\n                lifetime for.\n            logcontext:\n                the log context to which this scope is attached.\n            enter_logcontext:\n                if True the log context will be exited when the scope is finished\n            finish_on_close:\n                if True finish the span when the scope is closed\n        '
        super().__init__(manager, span)
        self.logcontext = logcontext
        self._finish_on_close = finish_on_close
        self._enter_logcontext = enter_logcontext

    def __exit__(self, exc_type: Optional[Type[BaseException]], value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        if False:
            print('Hello World!')
        if exc_type == twisted.internet.defer._DefGen_Return:
            exc_type = value = traceback = None
        super().__exit__(exc_type, value, traceback)

    def __str__(self) -> str:
        if False:
            return 10
        return f'Scope<{self.span}>'

    def close(self) -> None:
        if False:
            i = 10
            return i + 15
        active_scope = self.manager.active
        if active_scope is not self:
            logger.error('Closing scope %s which is not the currently-active one %s', self, active_scope)
        if self._finish_on_close:
            self.span.finish()
        self.logcontext.scope = None
        if self._enter_logcontext:
            self.logcontext.__exit__(None, None, None)