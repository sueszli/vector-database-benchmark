"""Here we run tests directly with SimpleUpdateProcessor because that's easier than providing dummy
implementations for SimpleUpdateProcessor and we want to test SimpleUpdateProcessor anyway."""
import asyncio
import pytest
from telegram import Update
from telegram.ext import SimpleUpdateProcessor
from tests.auxil.asyncio_helpers import call_after
from tests.auxil.slots import mro_slots

@pytest.fixture()
def mock_processor():
    if False:
        return 10

    class MockProcessor(SimpleUpdateProcessor):
        test_flag = False

        async def do_process_update(self, update, coroutine):
            await coroutine
            self.test_flag = True
    return MockProcessor(5)

class TestSimpleUpdateProcessor:

    def test_slot_behaviour(self):
        if False:
            for i in range(10):
                print('nop')
        inst = SimpleUpdateProcessor(1)
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    @pytest.mark.parametrize('concurrent_updates', [0, -1])
    def test_init(self, concurrent_updates):
        if False:
            i = 10
            return i + 15
        processor = SimpleUpdateProcessor(3)
        assert processor.max_concurrent_updates == 3
        with pytest.raises(ValueError, match='must be a positive integer'):
            SimpleUpdateProcessor(concurrent_updates)

    async def test_process_update(self, mock_processor):
        """Test that process_update calls do_process_update."""
        update = Update(1)

        async def coroutine():
            pass
        await mock_processor.process_update(update, coroutine())
        assert mock_processor.test_flag

    async def test_do_process_update(self):
        """Test that do_process_update calls the coroutine."""
        processor = SimpleUpdateProcessor(1)
        update = Update(1)
        test_flag = False

        async def coroutine():
            nonlocal test_flag
            test_flag = True
        await processor.do_process_update(update, coroutine())
        assert test_flag

    async def test_max_concurrent_updates_enforcement(self, mock_processor):
        """Test that max_concurrent_updates is enforced, i.e. that the processor will run
        at most max_concurrent_updates coroutines at the same time."""
        count = 2 * mock_processor.max_concurrent_updates
        events = {i: asyncio.Event() for i in range(count)}
        queue = asyncio.Queue()
        for event in events.values():
            await queue.put(event)

        async def callback():
            await asyncio.sleep(0.5)
            (await queue.get()).set()
        tasks = [asyncio.create_task(mock_processor.process_update(update=_, coroutine=callback())) for _ in range(count)]
        for i in range(count):
            assert not events[i].is_set()
        await asyncio.sleep(0.75)
        for i in range(mock_processor.max_concurrent_updates):
            assert events[i].is_set()
        for i in range(mock_processor.max_concurrent_updates, count):
            assert not events[i].is_set()
        await asyncio.sleep(0.5)
        for i in range(count):
            assert events[i].is_set()
        await asyncio.gather(*tasks)

    async def test_context_manager(self, monkeypatch, mock_processor):
        self.test_flag = set()

        async def after_initialize(*args, **kwargs):
            self.test_flag.add('initialize')

        async def after_shutdown(*args, **kwargs):
            self.test_flag.add('stop')
        monkeypatch.setattr(SimpleUpdateProcessor, 'initialize', call_after(SimpleUpdateProcessor.initialize, after_initialize))
        monkeypatch.setattr(SimpleUpdateProcessor, 'shutdown', call_after(SimpleUpdateProcessor.shutdown, after_shutdown))
        async with mock_processor:
            pass
        assert self.test_flag == {'initialize', 'stop'}

    async def test_context_manager_exception_on_init(self, monkeypatch, mock_processor):

        async def initialize(*args, **kwargs):
            raise RuntimeError('initialize')

        async def shutdown(*args, **kwargs):
            self.test_flag = 'shutdown'
        monkeypatch.setattr(SimpleUpdateProcessor, 'initialize', initialize)
        monkeypatch.setattr(SimpleUpdateProcessor, 'shutdown', shutdown)
        with pytest.raises(RuntimeError, match='initialize'):
            async with mock_processor:
                pass
        assert self.test_flag == 'shutdown'