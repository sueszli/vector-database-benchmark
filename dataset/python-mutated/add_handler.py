import pyrogram
from pyrogram.handlers import DisconnectHandler
from pyrogram.handlers.handler import Handler

class AddHandler:

    def add_handler(self: 'pyrogram.Client', handler: 'Handler', group: int=0):
        if False:
            return 10
        'Register an update handler.\n\n        You can register multiple handlers, but at most one handler within a group will be used for a single update.\n        To handle the same update more than once, register your handler using a different group id (lower group id\n        == higher priority). This mechanism is explained in greater details at\n        :doc:`More on Updates <../../topics/more-on-updates>`.\n\n        Parameters:\n            handler (``Handler``):\n                The handler to be registered.\n\n            group (``int``, *optional*):\n                The group identifier, defaults to 0.\n\n        Returns:\n            ``tuple``: A tuple consisting of *(handler, group)*.\n\n        Example:\n            .. code-block:: python\n\n                from pyrogram import Client\n                from pyrogram.handlers import MessageHandler\n\n                async def hello(client, message):\n                    print(message)\n\n                app = Client("my_account")\n\n                app.add_handler(MessageHandler(hello))\n\n                app.run()\n        '
        if isinstance(handler, DisconnectHandler):
            self.disconnect_handler = handler.callback
        else:
            self.dispatcher.add_handler(handler, group)
        return (handler, group)