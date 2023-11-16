import asyncio
import sys

def load_loop_functions():
    if False:
        return 10
    if sys.version_info >= (3, 7):

        def get_task_loop(task):
            if False:
                for i in range(10):
                    print('nop')
            return task.get_loop()
        get_running_loop = asyncio.get_running_loop
    else:

        def get_task_loop(task):
            if False:
                while True:
                    i = 10
            return task._loop

        def get_running_loop():
            if False:
                i = 10
                return i + 15
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                raise RuntimeError('There is no running event loop')
            return loop
    return (get_task_loop, get_running_loop)
(get_task_loop, get_running_loop) = load_loop_functions()