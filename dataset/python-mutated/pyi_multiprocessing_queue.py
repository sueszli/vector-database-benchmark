import sys
import multiprocessing

class SendEventProcess(multiprocessing.Process):

    def __init__(self, queue):
        if False:
            return 10
        multiprocessing.Process.__init__(self)
        self.queue = queue

    def run(self):
        if False:
            return 10
        print('SendEventProcess: begin')
        self.queue.put((1, 2))
        print('SendEventProcess: end')

def main(start_method):
    if False:
        while True:
            i = 10
    multiprocessing.set_start_method(start_method)
    print('Main: begin')
    queue = multiprocessing.Queue()
    process = SendEventProcess(queue)
    process.start()
    results = queue.get()
    print(f'Main: retrieved results: {results}')
    assert results == (1, 2)
    process.join()
    print('Main: end')
    assert process.exitcode == 0, f'Process exited with non-success code {process.exitcode}!'
if __name__ == '__main__':
    multiprocessing.freeze_support()
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <start-method>')
        sys.exit(1)
    main(sys.argv[1])