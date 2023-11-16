from __future__ import annotations
import logging
import sys
import warnings
import weakref
from typing import TYPE_CHECKING, NoReturn
import attr
from .. import _core
from .._util import name_asyncgen
from . import _run
ASYNCGEN_LOGGER = logging.getLogger('trio.async_generator_errors')
if TYPE_CHECKING:
    from types import AsyncGeneratorType
    from typing import Set
    _WEAK_ASYNC_GEN_SET = weakref.WeakSet[AsyncGeneratorType[object, NoReturn]]
    _ASYNC_GEN_SET = Set[AsyncGeneratorType[object, NoReturn]]
else:
    _WEAK_ASYNC_GEN_SET = weakref.WeakSet
    _ASYNC_GEN_SET = set

@attr.s(eq=False, slots=True)
class AsyncGenerators:
    alive: _WEAK_ASYNC_GEN_SET | _ASYNC_GEN_SET = attr.ib(factory=_WEAK_ASYNC_GEN_SET)
    trailing_needs_finalize: _ASYNC_GEN_SET = attr.ib(factory=_ASYNC_GEN_SET)
    prev_hooks = attr.ib(init=False)

    def install_hooks(self, runner: _run.Runner) -> None:
        if False:
            return 10

        def firstiter(agen: AsyncGeneratorType[object, NoReturn]) -> None:
            if False:
                while True:
                    i = 10
            if hasattr(_run.GLOBAL_RUN_CONTEXT, 'task'):
                self.alive.add(agen)
            else:
                agen.ag_frame.f_locals['@trio_foreign_asyncgen'] = True
                if self.prev_hooks.firstiter is not None:
                    self.prev_hooks.firstiter(agen)

        def finalize_in_trio_context(agen: AsyncGeneratorType[object, NoReturn], agen_name: str) -> None:
            if False:
                for i in range(10):
                    print('nop')
            try:
                runner.spawn_system_task(self._finalize_one, agen, agen_name, name=f'close asyncgen {agen_name} (abandoned)')
            except RuntimeError:
                self.trailing_needs_finalize.add(agen)

        def finalizer(agen: AsyncGeneratorType[object, NoReturn]) -> None:
            if False:
                i = 10
                return i + 15
            agen_name = name_asyncgen(agen)
            try:
                is_ours = not agen.ag_frame.f_locals.get('@trio_foreign_asyncgen')
            except AttributeError:
                is_ours = True
            if is_ours:
                runner.entry_queue.run_sync_soon(finalize_in_trio_context, agen, agen_name)
                warnings.warn(f"Async generator {agen_name!r} was garbage collected before it had been exhausted. Surround its use in 'async with aclosing(...):' to ensure that it gets cleaned up as soon as you're done using it.", ResourceWarning, stacklevel=2, source=agen)
            elif self.prev_hooks.finalizer is not None:
                self.prev_hooks.finalizer(agen)
            else:
                closer = agen.aclose()
                try:
                    closer.send(None)
                except StopIteration:
                    pass
                else:
                    raise RuntimeError(f"Non-Trio async generator {agen_name!r} awaited something during finalization; install a finalization hook to support this, or wrap it in 'async with aclosing(...):'")
        self.prev_hooks = sys.get_asyncgen_hooks()
        sys.set_asyncgen_hooks(firstiter=firstiter, finalizer=finalizer)

    async def finalize_remaining(self, runner: _run.Runner) -> None:
        assert _core.current_task() is runner.init_task
        assert len(runner.tasks) == 2
        self.alive = set(self.alive)
        runner.entry_queue.run_sync_soon(runner.reschedule, runner.init_task)
        await _core.wait_task_rescheduled(lambda _: _core.Abort.FAILED)
        self.alive.update(self.trailing_needs_finalize)
        self.trailing_needs_finalize.clear()
        while self.alive:
            batch = self.alive
            self.alive = _ASYNC_GEN_SET()
            for agen in batch:
                await self._finalize_one(agen, name_asyncgen(agen))

    def close(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        sys.set_asyncgen_hooks(*self.prev_hooks)

    async def _finalize_one(self, agen: AsyncGeneratorType[object, NoReturn], name: object) -> None:
        try:
            with _core.CancelScope(shield=True) as cancel_scope:
                cancel_scope.cancel()
                await agen.aclose()
        except BaseException:
            ASYNCGEN_LOGGER.exception("Exception ignored during finalization of async generator %r -- surround your use of the generator in 'async with aclosing(...):' to raise exceptions like this in the context where they're generated", name)