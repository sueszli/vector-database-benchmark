import sys
import threading
import multiprocessing
NUM_THREADS = 4

def process_function(queue, i):
    if False:
        return 10
    print(f'Running process function with i={i}')
    queue.put(i)

def thread_function(queue, i):
    if False:
        i = 10
        return i + 15
    process = multiprocessing.Process(target=process_function, args=(queue, i))
    process.start()
    process.join()
    assert process.exitcode == 0, f'Process {i} exited with non-succcess code {process.exitcode}!'

def main(start_method):
    if False:
        print('Hello World!')
    multiprocessing.set_start_method(start_method)
    queue = multiprocessing.Queue()
    threads = []
    for i in range(NUM_THREADS):
        threads.append(threading.Thread(target=thread_function, args=(queue, i), daemon=True))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    results = []
    while not queue.empty():
        results.append(queue.get())
    print(f'Results: {results}')
    assert sorted(results) == list(range(NUM_THREADS))
if __name__ == '__main__':
    multiprocessing.freeze_support()
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <start-method>')
        sys.exit(1)
    main(sys.argv[1])