import concurrent.futures
from platform import python_version
import time
import uuid
from test.picardtestcase import PicardTestCase
from picard.util import pipe

def pipe_listener(pipe_handler):
    if False:
        return 10
    while True:
        for message in pipe_handler.read_from_pipe():
            if message and message != pipe.Pipe.NO_RESPONSE_MESSAGE:
                return message

class TestPipe(PicardTestCase):
    NAME = str(uuid.uuid4())
    VERSION = python_version()

    def test_invalid_args(self):
        if False:
            while True:
                i = 10
        self.assertRaises(pipe.PipeErrorInvalidArgs, pipe.Pipe, self.NAME, self.VERSION, 1)
        self.assertRaises(pipe.PipeErrorInvalidAppData, pipe.Pipe, 21, self.VERSION, None)
        self.assertRaises(pipe.PipeErrorInvalidAppData, pipe.Pipe, self.NAME, 21, None)

    def test_pipe_protocol(self):
        if False:
            while True:
                i = 10
        message = 'foo'
        __pool = concurrent.futures.ThreadPoolExecutor()
        pipe_handler = pipe.Pipe(self.NAME, self.VERSION)
        try:
            plistener = __pool.submit(pipe_listener, pipe_handler)
            time.sleep(0.2)
            res = ''
            try:
                pipe_handler.send_to_pipe(message)
                res = plistener.result(timeout=6)
            except concurrent.futures._base.TimeoutError:
                pass
            self.assertEqual(res, message, 'Data is sent and read correctly')
        finally:
            time.sleep(0.2)
            pipe_handler.stop()
            __pool.shutdown()