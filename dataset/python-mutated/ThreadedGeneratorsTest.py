import threading

def some_generator():
    if False:
        return 10
    yield 1

def run():
    if False:
        print('Hello World!')
    for i in range(10000):
        for j in some_generator():
            pass

def main():
    if False:
        for i in range(10):
            print('nop')
    workers = [threading.Thread(target=run) for i in range(5)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
if __name__ == '__main__':
    main()