"""
python_context.py by xianhu
"""
import contextlib

class MyOpen(object):

    def __init__(self, file_name):
        if False:
            for i in range(10):
                print('nop')
        '初始化方法'
        self.file_name = file_name
        self.file_handler = None
        return

    def __enter__(self):
        if False:
            while True:
                i = 10
        'enter方法，返回file_handler'
        print('enter:', self.file_name)
        self.file_handler = open(self.file_name, 'r')
        return self.file_handler

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            return 10
        'exit方法，关闭文件并返回True'
        print('exit:', exc_type, exc_val, exc_tb)
        if self.file_handler:
            self.file_handler.close()
        return True
with MyOpen('python_base.py') as file_in:
    for line in file_in:
        print(line)
        raise ZeroDivisionError

@contextlib.contextmanager
def open_func(file_name):
    if False:
        return 10
    print('open file:', file_name, 'in __enter__')
    file_handler = open(file_name, 'r')
    yield file_handler
    print('close file:', file_name, 'in __exit__')
    file_handler.close()
    return
with open_func('python_base.py') as file_in:
    for line in file_in:
        print(line)
        break

class MyOpen2(object):

    def __init__(self, file_name):
        if False:
            for i in range(10):
                print('nop')
        '初始化方法'
        self.file_handler = open(file_name, 'r')
        return

    def close(self):
        if False:
            i = 10
            return i + 15
        '关闭文件，会被自动调用'
        print('call close in MyOpen2')
        if self.file_handler:
            self.file_handler.close()
        return
with contextlib.closing(MyOpen2('python_base.py')) as file_in:
    pass