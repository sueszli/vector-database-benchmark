"""
Make events hooks non-blocking using async or @concurrent.
"""
import asyncio
import logging
import time
from mitmproxy.script import concurrent
if True:

    async def request(flow):
        logging.info(f'handle request: {flow.request.host}{flow.request.path}')
        await asyncio.sleep(5)
        logging.info(f'start  request: {flow.request.host}{flow.request.path}')
else:

    @concurrent
    def request(flow):
        if False:
            for i in range(10):
                print('nop')
        logging.info(f'handle request: {flow.request.host}{flow.request.path}')
        time.sleep(5)
        logging.info(f'start  request: {flow.request.host}{flow.request.path}')