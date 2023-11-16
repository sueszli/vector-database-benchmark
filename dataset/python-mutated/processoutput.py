import asyncio
from contextlib import suppress
from typing import Callable, List, Optional

class ProcessOutput:

    def __init__(self, command: List[str], timeout: Optional[float]=None):
        if False:
            return 10
        self.command = command
        self.timeout = timeout

    def run(self) -> bool:
        if False:
            i = 10
            return i + 15
        return asyncio.run(self._run())

    async def _run(self) -> bool:
        loop = asyncio.get_event_loop()
        done: asyncio.Future[bool] = loop.create_future()
        process = await asyncio.create_subprocess_exec(*self.command, stdin=None, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        if not process.stdout or not process.stderr:
            return False

        async def ontimeout():
            if self.timeout:
                await asyncio.sleep(self.timeout)
                done.set_result(False)

        async def onexit():
            code = await process.wait()
            done.set_result(self.onexit(code))

        async def onoutput(callback: Callable[[int, str], Optional[bool]], streamreader: asyncio.StreamReader):
            line: bytes
            idx = 0
            async for line in streamreader:
                try:
                    result = callback(idx, line.decode().rstrip())
                except Exception as err:
                    done.set_exception(err)
                    break
                if result is not None:
                    done.set_result(bool(result))
                    break
                idx += 1
        tasks = (loop.create_task(ontimeout()), loop.create_task(onoutput(self.onstdout, process.stdout)), loop.create_task(onoutput(self.onstderr, process.stderr)), loop.create_task(onexit()))
        try:
            return await done
        finally:
            for task in tasks:
                task.cancel()
            with suppress(OSError):
                process.kill()

    def onexit(self, code: int) -> bool:
        if False:
            return 10
        return code == 0

    def onstdout(self, idx: int, line: str) -> Optional[bool]:
        if False:
            while True:
                i = 10
        pass

    def onstderr(self, idx: int, line: str) -> Optional[bool]:
        if False:
            i = 10
            return i + 15
        pass