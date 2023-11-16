class MyDB:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.connection = Connection()

    def connect(self, connection_string):
        if False:
            i = 10
            return i + 15
        return self.connection

class Connection:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.cur = Cursor()

    def cursor(self):
        if False:
            print('Hello World!')
        return self.cur

    def close(self):
        if False:
            while True:
                i = 10
        pass

class Cursor:

    def execute(self, query):
        if False:
            return 10
        if query == 'select id from employee_db where name=John':
            return 123
        elif query == 'select id from employee_db where name=Tom':
            return 789
        else:
            return -1

    def close(self):
        if False:
            print('Hello World!')
        pass