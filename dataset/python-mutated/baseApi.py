__author__ = 'mayanqiong'
import asyncio
import functools
import sys
import time
from asyncio import Future
from typing import Optional, Coroutine

class TqBaseApi(object):
    """
    天勤基类，处理 EventLoop 相关的调用
    """

    def __init__(self, loop: Optional[asyncio.AbstractEventLoop]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        创建天勤接口实例\n\n        Args:\n            loop(asyncio.AbstractEventLoop): [可选] 使用指定的 IOLoop, 默认创建一个新的.\n        '
        self._loop = asyncio.SelectorEventLoop() if loop is None else loop
        (self._event_rev, self._check_rev) = (0, 0)
        self._wait_idle_list = []
        self._wait_timeout = False
        self._tasks = set()
        self._exceptions = []
        self._loop.call_soon = functools.partial(self._call_soon, self._loop.call_soon)
        if sys.platform.startswith('win'):
            self._create_task(self._windows_patch())

    def _create_task(self, coro: Coroutine, _caller_api: bool=False) -> asyncio.Task:
        if False:
            i = 10
            return i + 15
        task = self._loop.create_task(coro)
        py_ver = sys.version_info
        current_task = asyncio.Task.current_task(loop=self._loop) if py_ver.major == 3 and py_ver.minor < 7 else asyncio.current_task(loop=self._loop)
        if current_task is None or _caller_api:
            self._tasks.add(task)
            task.add_done_callback(self._on_task_done)
        return task

    def _call_soon(self, org_call_soon, callback, *args, **kargs):
        if False:
            for i in range(10):
                print('nop')
        'ioloop.call_soon的补丁, 用来追踪是否有任务完成并等待执行'
        self._event_rev += 1
        return org_call_soon(callback, *args, **kargs)

    def _run_once(self):
        if False:
            for i in range(10):
                print('nop')
        '执行 ioloop 直到 ioloop.stop 被调用'
        if not self._exceptions:
            self._loop.run_forever()
        if self._exceptions:
            raise self._exceptions.pop(0)

    def _run_until_idle(self, async_run=False):
        if False:
            return 10
        '执行 ioloop 直到没有待执行任务\n        async_run is True 会从 _wait_idle_list 中取出等待的异步任务，保证同步代码优先于异步代码执行，\n        只有被 _run_until_task_done 调用（即 api 等待 fetch_msg）时，async_run 会为 True\n        '
        while self._check_rev != self._event_rev:
            check_handle = self._loop.call_soon(self._check_event, self._event_rev + 1)
            try:
                self._run_once()
            finally:
                check_handle.cancel()
        if len(self._wait_idle_list) > 0 and async_run:
            f = self._wait_idle_list.pop(0)
            f.set_result(None)

    async def _wait_until_idle(self):
        """等待 ioloop 执行到空闲时，才从网络连接处收数据包，在 TqConnect 类中使用"""
        f = Future()
        self._wait_idle_list.append(f)
        self._loop.stop()
        await f

    def _run_until_task_done(self, task: asyncio.Task, deadline=None):
        if False:
            return 10
        try:
            self._wait_timeout = False
            if deadline is not None:
                deadline_handle = self._loop.call_later(max(0, deadline - time.time()), self._set_wait_timeout)
            while not self._wait_timeout and (not task.done()):
                if len(self._wait_idle_list) == 0:
                    self._run_once()
                else:
                    self._run_until_idle(async_run=True)
        finally:
            if deadline is not None:
                deadline_handle.cancel()
            task.cancel()

    def _check_event(self, rev):
        if False:
            for i in range(10):
                print('nop')
        self._check_rev = rev
        self._loop.stop()

    def _set_wait_timeout(self):
        if False:
            i = 10
            return i + 15
        self._wait_timeout = True
        self._loop.stop()

    def _on_task_done(self, task):
        if False:
            return 10
        '当由 api 维护的 task 执行完成后取出运行中遇到的例外并停止 ioloop'
        try:
            exception = task.exception()
            if exception:
                self._exceptions.append(exception)
        except asyncio.CancelledError:
            pass
        finally:
            self._tasks.remove(task)
            self._loop.stop()

    async def _windows_patch(self):
        """Windows系统下asyncio不支持KeyboardInterrupt的临时补丁, 详见 https://bugs.python.org/issue23057"""
        while True:
            await asyncio.sleep(1)

    def _close(self) -> None:
        if False:
            while True:
                i = 10
        self._run_until_idle(async_run=False)
        for task in self._tasks:
            task.cancel()
        while self._tasks:
            self._run_once()
        self._loop.run_until_complete(self._loop.shutdown_asyncgens())
        self._loop.close()