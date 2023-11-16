import pyrogram
from pyrogram.handlers import DisconnectHandler
from pyrogram.handlers.handler import Handler

class RemoveHandler:

    def remove_handler(self: 'pyrogram.Client', handler: 'Handler', group: int=0):
        if False:
            return 10
        'Remove a previously-registered update handler.\n\n        Make sure to provide the right group where the handler was added in. You can use the return value of the\n        :meth:`~pyrogram.Client.add_handler` method, a tuple of *(handler, group)*, and pass it directly.\n\n        Parameters:\n            handler (``Handler``):\n                The handler to be removed.\n\n            group (``int``, *optional*):\n                The group identifier, defaults to 0.\n\n        Example:\n            .. code-block:: python\n\n                from pyrogram import Client\n                from pyrogram.handlers import MessageHandler\n\n                async def hello(client, message):\n                    print(message)\n\n                app = Client("my_account")\n\n                handler = app.add_handler(MessageHandler(hello))\n\n                # Starred expression to unpack (handler, group)\n                app.remove_handler(*handler)\n\n                app.run()\n        '
        if isinstance(handler, DisconnectHandler):
            self.disconnect_handler = None
        else:
            self.dispatcher.remove_handler(handler, group)