import pytest_twisted
from twisted.internet.task import deferLater

def sleep():
    if False:
        i = 10
        return i + 15
    import twisted.internet.reactor
    return deferLater(clock=twisted.internet.reactor, delay=0)

@pytest_twisted.inlineCallbacks
def test_inlineCallbacks():
    if False:
        print('Hello World!')
    yield sleep()

@pytest_twisted.ensureDeferred
async def test_inlineCallbacks_async():
    await sleep()