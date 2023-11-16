def myfunc():
    if False:
        for i in range(10):
            print('nop')
    print({"also won't be reformatted"})

def myfunc():
    if False:
        print('Hello World!')
    print({"also won't be reformatted"})

def myfunc():
    if False:
        return 10
    print({"also won't be reformatted"})

def myfunc():
    if False:
        while True:
            i = 10
    print({'this will be reformatted'})

def myfunc():
    if False:
        return 10
    print({"also won't be reformatted"})

def myfunc():
    if False:
        for i in range(10):
            print('nop')
    print({"also won't be reformatted"})

def myfunc():
    if False:
        i = 10
        return i + 15
    print({"also won't be reformatted"})

def myfunc():
    if False:
        for i in range(10):
            print('nop')
    print({'this will be reformatted'})