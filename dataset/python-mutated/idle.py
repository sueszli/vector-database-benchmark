import asyncio
import logging
import signal
from signal import signal as signal_fn, SIGINT, SIGTERM, SIGABRT
log = logging.getLogger(__name__)
signals = {k: v for (v, k) in signal.__dict__.items() if v.startswith('SIG') and (not v.startswith('SIG_'))}

async def idle():
    """Block the main script execution until a signal is received.

    This function will run indefinitely in order to block the main script execution and prevent it from
    exiting while having client(s) that are still running in the background.

    It is useful for event-driven application only, that are, applications which react upon incoming Telegram
    updates through handlers, rather than executing a set of methods sequentially.

    Once a signal is received (e.g.: from CTRL+C) the function will terminate and your main script will continue.
    Don't forget to call :meth:`~pyrogram.Client.stop` for each running client before the script ends.

    Example:
        .. code-block:: python

            import asyncio
            from pyrogram import Client, idle


            async def main():
                apps = [
                    Client("account1"),
                    Client("account2"),
                    Client("account3")
                ]

                ...  # Set up handlers

                for app in apps:
                    await app.start()

                await idle()

                for app in apps:
                    await app.stop()


            asyncio.run(main())
    """
    task = None

    def signal_handler(signum, __):
        if False:
            return 10
        logging.info(f'Stop signal received ({signals[signum]}). Exiting...')
        task.cancel()
    for s in (SIGINT, SIGTERM, SIGABRT):
        signal_fn(s, signal_handler)
    while True:
        task = asyncio.create_task(asyncio.sleep(600))
        try:
            await task
        except asyncio.CancelledError:
            break