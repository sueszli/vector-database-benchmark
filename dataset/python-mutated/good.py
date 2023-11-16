from tornado import gen
from tornado.web import RequestHandler

class MyHandler(RequestHandler):

    def get(self) -> None:
        if False:
            i = 10
            return i + 15
        self.write('foo')

    async def post(self) -> None:
        await gen.sleep(1)
        self.write('foo')