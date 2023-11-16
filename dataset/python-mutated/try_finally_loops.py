while True:
    try:
        break
    finally:
        print('finally 1')
for i in [1, 5, 10]:
    try:
        continue
    finally:
        print('finally 2')
for i in range(3):
    try:
        continue
    finally:
        print('finally 3')
for i in range(4):
    print(i)
    try:
        while True:
            try:
                try:
                    break
                finally:
                    print('finally 1')
            finally:
                print('finally 2')
        print('here')
    finally:
        print('finnaly 3')
for i in [1]:
    try:
        print(i)
        break
    finally:
        print('finally 4')
for i in [1]:
    try:
        break
    finally:
        pass

def f():
    if False:
        return 10
    for i in [1]:
        try:
            break
        finally:
            pass

def g():
    if False:
        print('Hello World!')
    global global_var
    f()
    print(global_var)
global_var = 'global'
g()