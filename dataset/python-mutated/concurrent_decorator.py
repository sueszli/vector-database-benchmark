import time
from mitmproxy.script import concurrent

@concurrent
def request(flow):
    if False:
        print('Hello World!')
    time.sleep(0.25)

@concurrent
async def requestheaders(flow):
    time.sleep(0.25)