"""Add a custom command to mitmproxy's command prompt."""
import logging
from mitmproxy import command

class MyAddon:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.num = 0

    @command.command('myaddon.inc')
    def inc(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.num += 1
        logging.info(f'num = {self.num}')
addons = [MyAddon()]