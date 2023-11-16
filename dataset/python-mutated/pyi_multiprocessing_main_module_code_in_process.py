import sys
import multiprocessing

def process_function():
    if False:
        i = 10
        return i + 15
    import dis
    import sys
    main_code = sys.modules['__main__'].__loader__.get_code('__main__')
    print(dis.dis(main_code))

def main(start_method):
    if False:
        while True:
            i = 10
    multiprocessing.set_start_method(start_method)
    process = multiprocessing.Process(target=process_function)
    process.start()
    process.join()
    assert process.exitcode == 0, f'Process exited with non-success code {process.exitcode}!'
if __name__ == '__main__':
    multiprocessing.freeze_support()
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <start-method>')
        sys.exit(1)
    main(sys.argv[1])