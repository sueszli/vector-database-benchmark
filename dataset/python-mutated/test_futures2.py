import asyncio
import unittest

def tearDownModule():
    if False:
        i = 10
        return i + 15
    asyncio.set_event_loop_policy(None)

class FutureTests(unittest.IsolatedAsyncioTestCase):

    async def test_recursive_repr_for_pending_tasks(self):

        async def func():
            return asyncio.all_tasks()
        self.assertIn('...', repr(await asyncio.wait_for(func(), timeout=10)))
if __name__ == '__main__':
    unittest.main()