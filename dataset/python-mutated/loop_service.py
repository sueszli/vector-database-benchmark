"""
The base class for all Prefect REST API loop services.
"""
import asyncio
import signal
from typing import List
import anyio
import pendulum
from prefect.logging import get_logger
from prefect.server.database.dependencies import inject_db
from prefect.server.database.interface import PrefectDBInterface

class LoopService:
    """
    Loop services are relatively lightweight maintenance routines that need to run periodically.

    This class makes it straightforward to design and integrate them. Users only need to
    define the `run_once` coroutine to describe the behavior of the service on each loop.
    """
    loop_seconds = 60

    def __init__(self, loop_seconds: float=None, handle_signals: bool=True):
        if False:
            print('Hello World!')
        '\n        Args:\n            loop_seconds (float): if provided, overrides the loop interval\n                otherwise specified as a class variable\n            handle_signals (bool): if True (default), SIGINT and SIGTERM are\n                gracefully intercepted and shut down the running service.\n        '
        if loop_seconds:
            self.loop_seconds = loop_seconds
        self._should_stop = False
        self._is_running = False
        self.name = type(self).__name__
        self.logger = get_logger(f'server.services.{self.name.lower()}')
        if handle_signals:
            signal.signal(signal.SIGINT, self._stop)
            signal.signal(signal.SIGTERM, self._stop)

    @inject_db
    async def _on_start(self, db: PrefectDBInterface) -> None:
        """
        Called prior to running the service
        """
        self._should_stop = False
        self._is_running = True

    async def _on_stop(self) -> None:
        """
        Called after running the service
        """
        self._is_running = False

    async def start(self, loops=None) -> None:
        """
        Run the service `loops` time. Pass loops=None to run forever.

        Args:
            loops (int, optional): the number of loops to run before exiting.
        """
        await self._on_start()
        i = 0
        while not self._should_stop:
            start_time = pendulum.now('UTC')
            try:
                self.logger.debug(f'About to run {self.name}...')
                await self.run_once()
            except NotImplementedError as exc:
                raise exc from None
            except Exception as exc:
                self.logger.error(f'Unexpected error in: {repr(exc)}', exc_info=True)
            end_time = pendulum.now('UTC')
            if (end_time - start_time).total_seconds() > self.loop_seconds:
                self.logger.warning(f'{self.name} took {(end_time - start_time).total_seconds()} seconds to run, which is longer than its loop interval of {self.loop_seconds} seconds.')
            i += 1
            if loops is not None and i == loops:
                self.logger.debug(f'{self.name} exiting after {loops} loop(s).')
                await self.stop(block=False)
            next_run = max(start_time.add(seconds=self.loop_seconds), pendulum.now('UTC'))
            self.logger.debug(f'Finished running {self.name}. Next run at {next_run}')
            while pendulum.now('UTC') < next_run and (not self._should_stop):
                await asyncio.sleep(min(1, (next_run - pendulum.now('UTC')).total_seconds()))
        await self._on_stop()

    async def stop(self, block=True) -> None:
        """
        Gracefully stops a running LoopService and optionally blocks until the
        service stops.

        Args:
            block (bool): if True, blocks until the service is
                finished running. Otherwise it requests a stop and returns but
                the service may still be running a final loop.

        """
        self._stop()
        if block:
            with anyio.move_on_after(self.loop_seconds):
                while self._is_running:
                    await asyncio.sleep(0.1)
            if self._is_running:
                self.logger.warning(f'`stop(block=True)` was called on {self.name} but more than one loop interval ({self.loop_seconds} seconds) has passed. This usually means something is wrong. If `stop()` was called from inside the loop service, use `stop(block=False)` instead.')

    def _stop(self, *_) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Private, synchronous method for setting the `_should_stop` flag. Takes arbitrary\n        arguments so it can be used as a signal handler.\n        '
        self._should_stop = True

    async def run_once(self) -> None:
        """
        Represents one loop of the service.

        Users should override this method.

        To actually run the service once, call `LoopService().start(loops=1)`
        instead of `LoopService().run_once()`, because this method will not invoke setup
        and teardown methods properly.
        """
        raise NotImplementedError('LoopService subclasses must implement this method.')

async def run_multiple_services(loop_services: List[LoopService]):
    """
    Only one signal handler can be active at a time, so this function takes a list
    of loop services and runs all of them with a global signal handler.
    """

    def stop_all_services(self, *_):
        if False:
            for i in range(10):
                print('nop')
        for service in loop_services:
            service._stop()
    signal.signal(signal.SIGINT, stop_all_services)
    signal.signal(signal.SIGTERM, stop_all_services)
    await asyncio.gather(*[service.start() for service in loop_services])