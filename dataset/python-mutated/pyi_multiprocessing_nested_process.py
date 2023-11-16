import sys
import multiprocessing

def nested_process_function(queue):
    if False:
        while True:
            i = 10
    print('Running nested sub-process!')
    queue.put(2)

def process_function(queue):
    if False:
        print('Hello World!')
    print('Running sub-process!')
    queue.put(1)
    process = multiprocessing.Process(target=nested_process_function, args=(queue,))
    process.start()
    process.join()

def main(start_method):
    if False:
        print('Hello World!')
    multiprocessing.set_start_method(start_method)
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=process_function, args=(queue,))
    process.start()
    process.join()
    results = []
    while not queue.empty():
        results.append(queue.get())
    print(f'Results: {results}')
    assert results == [1, 2]
if __name__ == '__main__':
    multiprocessing.freeze_support()
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <start-method>')
        sys.exit(1)
    main(sys.argv[1])