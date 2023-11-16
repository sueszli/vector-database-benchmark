import os
import signal
import subprocess
import sys
import threading
import time
from typing import Any

def clean_and_terminate(task: 'subprocess.Popen') -> None:
    if False:
        print('Hello World!')
    task.terminate()
    time.sleep(0.5)
    if task.poll() is None:
        task.kill()

def check_parent_alive(task: 'subprocess.Popen') -> None:
    if False:
        print('Hello World!')
    orig_parent_id = os.getppid()
    while True:
        if os.getppid() != orig_parent_id:
            clean_and_terminate(task)
            break
        time.sleep(0.5)
if __name__ == '__main__':
    '\n    This is a wrapper around torch.distributed.run and it kills the child process\n    if the parent process fails, crashes, or exits.\n    '
    args = sys.argv[1:]
    cmd = [sys.executable, '-m', 'torch.distributed.run', *args]
    task = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.PIPE, env=os.environ)
    t = threading.Thread(target=check_parent_alive, args=(task,), daemon=True)

    def sigterm_handler(*args: Any) -> None:
        if False:
            while True:
                i = 10
        clean_and_terminate(task)
        os._exit(0)
    signal.signal(signal.SIGTERM, sigterm_handler)
    t.start()
    task.stdin.close()
    try:
        for line in task.stdout:
            decoded = line.decode()
            print(decoded.rstrip())
        task.wait()
    finally:
        if task.poll() is None:
            try:
                task.terminate()
                time.sleep(0.5)
                if task.poll() is None:
                    task.kill()
            except OSError:
                pass