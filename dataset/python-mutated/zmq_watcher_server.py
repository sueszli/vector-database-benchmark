from tensorwatch.watcher import Watcher
import time
from tensorwatch import utils
utils.set_debug_verbosity(10)

def main():
    if False:
        while True:
            i = 10
    watcher = Watcher()
    for i in range(5000):
        watcher.observe(x=i)
        time.sleep(1)
main()