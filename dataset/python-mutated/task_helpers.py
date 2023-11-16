import time
TOLERANCE = 2

def wait_till_task_finish(task):
    if False:
        i = 10
        return i + 15
    start = time.time()
    while task.is_alive():
        time.sleep(0.001)
        task.session.scheduler.handle_logs()
        if time.time() - start >= TOLERANCE:
            raise TimeoutError('Task did not finish.')
    task.session.scheduler.handle_logs()