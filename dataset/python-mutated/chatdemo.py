import asyncio
import tornado
import os.path
import uuid
from tornado.options import define, options, parse_command_line
define('port', default=8888, help='run on the given port', type=int)
define('debug', default=True, help='run in debug mode')

class MessageBuffer(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.cond = tornado.locks.Condition()
        self.cache = []
        self.cache_size = 200

    def get_messages_since(self, cursor):
        if False:
            for i in range(10):
                print('nop')
        'Returns a list of messages newer than the given cursor.\n\n        ``cursor`` should be the ``id`` of the last message received.\n        '
        results = []
        for msg in reversed(self.cache):
            if msg['id'] == cursor:
                break
            results.append(msg)
        results.reverse()
        return results

    def add_message(self, message):
        if False:
            while True:
                i = 10
        self.cache.append(message)
        if len(self.cache) > self.cache_size:
            self.cache = self.cache[-self.cache_size:]
        self.cond.notify_all()
global_message_buffer = MessageBuffer()

class MainHandler(tornado.web.RequestHandler):

    def get(self):
        if False:
            i = 10
            return i + 15
        self.render('index.html', messages=global_message_buffer.cache)

class MessageNewHandler(tornado.web.RequestHandler):
    """Post a new message to the chat room."""

    def post(self):
        if False:
            i = 10
            return i + 15
        message = {'id': str(uuid.uuid4()), 'body': self.get_argument('body')}
        message['html'] = tornado.escape.to_unicode(self.render_string('message.html', message=message))
        if self.get_argument('next', None):
            self.redirect(self.get_argument('next'))
        else:
            self.write(message)
        global_message_buffer.add_message(message)

class MessageUpdatesHandler(tornado.web.RequestHandler):
    """Long-polling request for new messages.

    Waits until new messages are available before returning anything.
    """

    async def post(self):
        cursor = self.get_argument('cursor', None)
        messages = global_message_buffer.get_messages_since(cursor)
        while not messages:
            self.wait_future = global_message_buffer.cond.wait()
            try:
                await self.wait_future
            except asyncio.CancelledError:
                return
            messages = global_message_buffer.get_messages_since(cursor)
        if self.request.connection.stream.closed():
            return
        self.write(dict(messages=messages))

    def on_connection_close(self):
        if False:
            for i in range(10):
                print('nop')
        self.wait_future.cancel()

async def main():
    parse_command_line()
    app = tornado.web.Application([('/', MainHandler), ('/a/message/new', MessageNewHandler), ('/a/message/updates', MessageUpdatesHandler)], cookie_secret='__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__', template_path=os.path.join(os.path.dirname(__file__), 'templates'), static_path=os.path.join(os.path.dirname(__file__), 'static'), xsrf_cookies=True, debug=options.debug)
    app.listen(options.port)
    await asyncio.Event().wait()
if __name__ == '__main__':
    asyncio.run(main())