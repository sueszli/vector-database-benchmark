from __future__ import annotations
import asyncio
from mitmproxy import ctx

class KeepServing:
    _watch_task: asyncio.Task | None = None

    def load(self, loader):
        if False:
            while True:
                i = 10
        loader.add_option('keepserving', bool, False, '\n            Continue serving after client playback, server playback or file\n            read. This option is ignored by interactive tools, which always keep\n            serving.\n            ')

    def keepgoing(self) -> bool:
        if False:
            return 10
        checks = ['readfile.reading', 'replay.client.count', 'replay.server.count']
        return any([ctx.master.commands.call(c) for c in checks])

    def shutdown(self):
        if False:
            for i in range(10):
                print('nop')
        ctx.master.shutdown()

    async def watch(self):
        while True:
            await asyncio.sleep(0.1)
            if not self.keepgoing():
                self.shutdown()

    def running(self):
        if False:
            for i in range(10):
                print('nop')
        opts = [ctx.options.client_replay, ctx.options.server_replay, ctx.options.rfile]
        if any(opts) and (not ctx.options.keepserving):
            self._watch_task = asyncio.get_running_loop().create_task(self.watch())