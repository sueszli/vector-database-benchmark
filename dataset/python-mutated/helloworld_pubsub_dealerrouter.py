"""Example using zmq with asyncio with pub/sub and dealer/router for asynchronous messages

Publisher sends either 'Hello World' or 'Hello Sekai' based on class language setting,
which is received by the Subscriber

When the Router receives a message from the Dealer, it changes the language setting"""
import asyncio
import logging
import traceback
import zmq
import zmq.asyncio
from zmq.asyncio import Context

class HelloWorld:

    def __init__(self) -> None:
        if False:
            return 10
        self.lang = 'eng'
        self.msg = 'Hello World'

    def change_language(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.lang == 'eng':
            self.lang = 'jap'
            self.msg = 'Hello Sekai'
        else:
            self.lang = 'eng'
            self.msg = 'Hello World'

    def msg_pub(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.msg

class HelloWorldPrinter:

    def msg_sub(self, msg: str) -> None:
        if False:
            while True:
                i = 10
        print(f'message received world: {msg}')

class HelloWorldMessage:

    def __init__(self, url: str='127.0.0.1', port: int=5555):
        if False:
            i = 10
            return i + 15
        print('Current libzmq version is %s' % zmq.zmq_version())
        print('Current  pyzmq version is %s' % zmq.__version__)
        self.url = f'tcp://{url}:{port}'
        self.ctx = Context.instance()
        self.hello_world = HelloWorld()

    def main(self) -> None:
        if False:
            i = 10
            return i + 15
        asyncio.run(asyncio.wait([self.hello_world_pub(), self.hello_world_sub(), self.lang_changer_router(), self.lang_changer_dealer()]))

    async def hello_world_pub(self) -> None:
        pub = self.ctx.socket(zmq.PUB)
        pub.connect(self.url)
        await asyncio.sleep(0.3)
        try:
            while True:
                msg = self.hello_world.msg_pub()
                print(f'world pub: {msg}')
                await asyncio.sleep(0.5)
                await pub.send_multipart([b'world', msg.encode('utf-8')])
        except Exception as e:
            print('Error with pub world')
            logging.error(traceback.format_exc())
            print()
        finally:
            pass

    async def hello_world_sub(self) -> None:
        print('Setting up world sub')
        obj = HelloWorldPrinter()
        sub = self.ctx.socket(zmq.SUB)
        sub.bind(self.url)
        sub.setsockopt(zmq.SUBSCRIBE, b'world')
        print('World sub initialized')
        try:
            while True:
                [topic, msg] = await sub.recv_multipart()
                print(f'world sub; topic: {topic.decode()}\tmessage: {msg.decode()}')
                obj.msg_sub(msg.decode('utf-8'))
        except Exception as e:
            print('Error with sub world')
            logging.error(traceback.format_exc())
            print()
        finally:
            pass

    async def lang_changer_dealer(self) -> None:
        deal = self.ctx.socket(zmq.DEALER)
        deal.setsockopt(zmq.IDENTITY, b'lang_dealer')
        deal.connect(self.url[:-1] + f'{int(self.url[-1]) + 1}')
        print('Command dealer initialized')
        await asyncio.sleep(0.3)
        msg = 'Change that language!'
        try:
            while True:
                print(f'Command deal: {msg}')
                await asyncio.sleep(2.0)
                await deal.send_multipart([msg.encode('utf-8')])
        except Exception as e:
            print('Error with pub world')
            logging.error(traceback.format_exc())
            print()
        finally:
            pass

    async def lang_changer_router(self) -> None:
        rout = self.ctx.socket(zmq.ROUTER)
        rout.bind(self.url[:-1] + f'{int(self.url[-1]) + 1}')
        print('Command router initialized')
        try:
            while True:
                [id_dealer, msg] = await rout.recv_multipart()
                print(f'Command rout; Sender ID: {id_dealer!r};\tmessage: {msg.decode()}')
                self.hello_world.change_language()
                print('Changed language! New language is: {}\n'.format(self.hello_world.lang))
        except Exception as e:
            print('Error with sub world')
            logging.error(traceback.format_exc())
            print()
        finally:
            pass

def main() -> None:
    if False:
        while True:
            i = 10
    hello_world = HelloWorldMessage()
    hello_world.main()
if __name__ == '__main__':
    main()