import time
from e2b import Sandbox
watcher = None

def create_watcher(sandbox):
    if False:
        for i in range(10):
            print('nop')
    watcher = sandbox.filesystem.watch_dir('/home')
    watcher.add_event_listener(lambda event: print(event))
    watcher.start()
sandbox = Sandbox(id='base')
create_watcher(sandbox)
for i in range(10):
    sandbox.filesystem.write(f'/home/file{i}.txt', f'Hello World {i}!')
    time.sleep(1)
sandbox.close()