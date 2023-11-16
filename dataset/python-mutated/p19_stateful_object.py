"""
Topic: 实现状态对象或状态机
Desc : 
"""

class Connection:
    """普通方案，好多个判断语句，效率低下~~"""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.state = 'CLOSED'

    def read(self):
        if False:
            print('Hello World!')
        if self.state != 'OPEN':
            raise RuntimeError('Not open')
        print('reading')

    def write(self, data):
        if False:
            for i in range(10):
                print('nop')
        if self.state != 'OPEN':
            raise RuntimeError('Not open')
        print('writing')

    def open(self):
        if False:
            i = 10
            return i + 15
        if self.state == 'OPEN':
            raise RuntimeError('Already open')
        self.state = 'OPEN'

    def close(self):
        if False:
            while True:
                i = 10
        if self.state == 'CLOSED':
            raise RuntimeError('Already closed')
        self.state = 'CLOSED'

class Connection1:
    """新方案——对每个状态定义一个类"""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.new_state(ClosedConnectionState)

    def new_state(self, newstate):
        if False:
            for i in range(10):
                print('nop')
        self._state = newstate

    def read(self):
        if False:
            print('Hello World!')
        return self._state.read(self)

    def write(self, data):
        if False:
            i = 10
            return i + 15
        return self._state.write(self, data)

    def open(self):
        if False:
            return 10
        return self._state.open(self)

    def close(self):
        if False:
            print('Hello World!')
        return self._state.close(self)

class ConnectionState:

    @staticmethod
    def read(conn):
        if False:
            return 10
        raise NotImplementedError()

    @staticmethod
    def write(conn, data):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    @staticmethod
    def open(conn):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    @staticmethod
    def close(conn):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

class ClosedConnectionState(ConnectionState):

    @staticmethod
    def read(conn):
        if False:
            return 10
        raise RuntimeError('Not open')

    @staticmethod
    def write(conn, data):
        if False:
            i = 10
            return i + 15
        raise RuntimeError('Not open')

    @staticmethod
    def open(conn):
        if False:
            for i in range(10):
                print('nop')
        conn.new_state(OpenConnectionState)

    @staticmethod
    def close(conn):
        if False:
            for i in range(10):
                print('nop')
        raise RuntimeError('Already closed')

class OpenConnectionState(ConnectionState):

    @staticmethod
    def read(conn):
        if False:
            for i in range(10):
                print('nop')
        print('reading')

    @staticmethod
    def write(conn, data):
        if False:
            for i in range(10):
                print('nop')
        print('writing')

    @staticmethod
    def open(conn):
        if False:
            i = 10
            return i + 15
        raise RuntimeError('Already open')

    @staticmethod
    def close(conn):
        if False:
            print('Hello World!')
        conn.new_state(ClosedConnectionState)
c = Connection1()
print(c._state)
c.open()

class State:

    def __init__(self):
        if False:
            return 10
        self.new_state(State_A)

    def new_state(self, state):
        if False:
            return 10
        self.__class__ = state

    def action(self, x):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

class State_A(State):

    def action(self, x):
        if False:
            return 10
        pass
        self.new_state(State_B)

class State_B(State):

    def action(self, x):
        if False:
            while True:
                i = 10
        pass
        self.new_state(State_C)

class State_C(State):

    def action(self, x):
        if False:
            for i in range(10):
                print('nop')
        pass
        self.new_state(State_A)