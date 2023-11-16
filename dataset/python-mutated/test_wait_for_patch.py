import asyncio

def test_wait_for_patched():
    if False:
        for i in range(10):
            print('nop')
    assert hasattr(asyncio.wait_for, 'patched')