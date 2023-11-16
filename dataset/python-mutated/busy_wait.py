import time

def function_1():
    if False:
        i = 10
        return i + 15
    pass

def function_2():
    if False:
        i = 10
        return i + 15
    pass

def main():
    if False:
        print('Hello World!')
    start_time = time.time()
    while time.time() < start_time + 0.25:
        function_1()
        function_2()
if __name__ == '__main__':
    main()