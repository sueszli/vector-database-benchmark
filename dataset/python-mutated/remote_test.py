import pytest
import threading
import time

@pytest.mark.asyncio
async def test_remote(df_remote, flush_guard):
    df = df_remote
    current_thread = threading.current_thread()
    called = False

    def progress(f):
        if False:
            return 10
        nonlocal called
        if f > 0:
            called = True
            assert threading.current_thread() is current_thread, 'the progress callback should not be invoked from the main thread'
        return True
    count_future = df.count(df.x, delay=True, progress=progress)
    await df.widget.execute_debounced()
    assert await count_future == 10
    assert called