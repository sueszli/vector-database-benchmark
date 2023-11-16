import trio

async def child1():
    print('  child1: started! sleeping now...')
    await trio.sleep(1)
    print('  child1: exiting!')

async def child2():
    print('  child2 started! sleeping now...')
    await trio.sleep(1)
    print('  child2 exiting!')

async def parent():
    print('parent: started!')
    async with trio.open_nursery() as nursery:
        print('parent: spawning child1...')
        nursery.start_soon(child1)
        print('parent: spawning child2...')
        nursery.start_soon(child2)
        print('parent: waiting for children to finish...')
    print('parent: all done!')

class Tracer(trio.abc.Instrument):

    def before_run(self):
        if False:
            i = 10
            return i + 15
        print('!!! run started')

    def _print_with_task(self, msg, task):
        if False:
            return 10
        print(f'{msg}: {task.name}')

    def task_spawned(self, task):
        if False:
            while True:
                i = 10
        self._print_with_task('### new task spawned', task)

    def task_scheduled(self, task):
        if False:
            for i in range(10):
                print('nop')
        self._print_with_task('### task scheduled', task)

    def before_task_step(self, task):
        if False:
            i = 10
            return i + 15
        self._print_with_task('>>> about to run one step of task', task)

    def after_task_step(self, task):
        if False:
            i = 10
            return i + 15
        self._print_with_task('<<< task step finished', task)

    def task_exited(self, task):
        if False:
            while True:
                i = 10
        self._print_with_task('### task exited', task)

    def before_io_wait(self, timeout):
        if False:
            return 10
        if timeout:
            print(f'### waiting for I/O for up to {timeout} seconds')
        else:
            print('### doing a quick check for I/O')
        self._sleep_time = trio.current_time()

    def after_io_wait(self, timeout):
        if False:
            for i in range(10):
                print('nop')
        duration = trio.current_time() - self._sleep_time
        print(f'### finished I/O check (took {duration} seconds)')

    def after_run(self):
        if False:
            while True:
                i = 10
        print('!!! run finished')
trio.run(parent, instruments=[Tracer()])