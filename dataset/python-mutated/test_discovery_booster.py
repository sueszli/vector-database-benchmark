import pytest
from tribler.core.components.ipv8.discovery_booster import DiscoveryBooster
TEST_BOOSTER_TIMEOUT_IN_SEC = 10
TEST_BOOSTER_TAKE_STEP_INTERVAL_IN_SEC = 1

@pytest.fixture(name='booster')
def fixture_booster():
    if False:
        i = 10
        return i + 15

    class MockWalker:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.take_step_called = False

        def take_step(self):
            if False:
                while True:
                    i = 10
            self.take_step_called = True
    return DiscoveryBooster(timeout_in_sec=TEST_BOOSTER_TIMEOUT_IN_SEC, take_step_interval_in_sec=TEST_BOOSTER_TAKE_STEP_INTERVAL_IN_SEC, walker=MockWalker())

@pytest.fixture(name='community')
def fixture_community():
    if False:
        while True:
            i = 10

    class MockCommunity:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.tasks = []

        def register_task(self, name, task, *args, delay=None, interval=None, ignore=()):
            if False:
                return 10
            self.tasks.append(name)

        def cancel_pending_task(self, name):
            if False:
                return 10
            self.tasks.remove(name)
    return MockCommunity()

def test_init(booster):
    if False:
        print('Hello World!')
    assert booster.timeout_in_sec == TEST_BOOSTER_TIMEOUT_IN_SEC
    assert booster.take_step_interval_in_sec == TEST_BOOSTER_TAKE_STEP_INTERVAL_IN_SEC
    assert booster.community is None
    assert booster.walker is not None

def test_apply(booster, community):
    if False:
        while True:
            i = 10
    booster.apply(None)
    assert booster.community is None
    booster.apply(community)
    assert booster.community == community
    assert booster.walker is not None
    assert len(community.tasks) == 2

def test_finish(booster, community):
    if False:
        print('Hello World!')
    booster.apply(community)
    booster.finish()
    assert len(community.tasks) == 1

def test_take_step(booster):
    if False:
        for i in range(10):
            print('nop')
    booster.take_step()
    assert booster.walker.take_step_called