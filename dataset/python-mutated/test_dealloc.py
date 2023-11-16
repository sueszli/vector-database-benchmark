import asyncio
import subprocess
import sys
from uvloop import _testbase as tb

class TestDealloc(tb.UVTestCase):

    def test_dealloc_1(self):
        if False:
            while True:
                i = 10

        async def test():
            prog = "import uvloop\n\nasync def foo():\n    return 42\n\ndef main():\n    loop = uvloop.new_event_loop()\n    loop.set_debug(True)\n    loop.run_until_complete(foo())\n    # Do not close the loop on purpose: let __dealloc__ methods run.\n\nif __name__ == '__main__':\n    main()\n            "
            cmd = sys.executable
            proc = await asyncio.create_subprocess_exec(cmd, b'-W', b'ignore', b'-c', prog, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            await proc.wait()
            out = await proc.stdout.read()
            err = await proc.stderr.read()
            return (out, err)
        (out, err) = self.loop.run_until_complete(test())
        self.assertEqual(out, b'', 'stdout is not empty')
        self.assertEqual(err, b'', 'stderr is not empty')