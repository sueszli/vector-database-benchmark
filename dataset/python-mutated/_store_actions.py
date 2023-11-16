import enum
import threading
from cupyx.distributed import _klv_utils

class Actions(enum.IntEnum):
    Set = 1
    Get = 2
    Barrier = 3

class ActionError:

    def __init__(self, exception):
        if False:
            for i in range(10):
                print('nop')
        self._exception = exception

    def klv(self):
        if False:
            for i in range(10):
                print('nop')
        e = self._exception
        return _klv_utils.get_result_action_t(1, str(e).encode('ascii'))

    @staticmethod
    def from_klv(klv):
        if False:
            i = 10
            return i + 15
        raise RuntimeError(klv._exception.decode('utf-8'))

    def decode_result(self, data):
        if False:
            i = 10
            return i + 15
        ActionError.from_klv(data)

def execute_action(action, value, store):
    if False:
        return 10
    try:
        if action == Actions.Set:
            action_obj = Set.from_klv(value)
        elif action == Actions.Get:
            action_obj = Get.from_klv(value)
        elif action == Actions.Barrier:
            action_obj = Barrier.from_klv(value)
        else:
            raise ValueError(f'unknown action {action}')
        return action_obj(store)
    except Exception as e:
        return ActionError(e)

class Set:

    class SetResult:

        def klv(self):
            if False:
                for i in range(10):
                    print('nop')
            v = bytearray(bytes(True))
            action = _klv_utils.get_result_action_t(0, v)
            return bytes(action)

        @staticmethod
        def from_klv(klv):
            if False:
                while True:
                    i = 10
            return True

    def __init__(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        self.key = key
        self.value = value
        if not isinstance(key, str):
            raise ValueError('Invalid type for key, only str allowed')
        if type(value) not in (bytes, bytearray, int):
            raise ValueError('Invalid type for value, only int or bytes allowed')

    @staticmethod
    def from_klv(value):
        if False:
            print('Hello World!')
        value = bytes(value)
        for (i, b) in enumerate(value):
            if b == 0:
                k = value[:i].decode('utf-8')
                value = value[i + 1:]
                break
        else:
            raise ValueError('No separation character for key found')
        v = _klv_utils.get_value_from_bytes(value)
        return Set(k, v)

    def klv(self):
        if False:
            while True:
                i = 10
        v = bytearray(self.key.encode('ascii'))
        v.append(0)
        v += _klv_utils.create_value_bytes(self.value)
        action = _klv_utils.get_action_t(Actions.Set, v)
        return bytes(action)

    def __call__(self, store):
        if False:
            print('Hello World!')
        store.storage[self.key] = self.value
        return Set.SetResult()

    def decode_result(self, data):
        if False:
            i = 10
            return i + 15
        return Set.SetResult.from_klv(data)

class Get:

    class GetResult:

        def __init__(self, value):
            if False:
                for i in range(10):
                    print('nop')
            self.value = value

        def klv(self):
            if False:
                for i in range(10):
                    print('nop')
            v = _klv_utils.create_value_bytes(self.value)
            action = _klv_utils.get_result_action_t(0, v)
            return bytes(action)

        @staticmethod
        def from_klv(value):
            if False:
                while True:
                    i = 10
            value = bytearray(value)
            return _klv_utils.get_value_from_bytes(value)

    def __init__(self, key):
        if False:
            for i in range(10):
                print('nop')
        self.key = key
        if not isinstance(key, str):
            raise ValueError('Invalid type for key, only str allowed')

    @staticmethod
    def from_klv(value):
        if False:
            print('Hello World!')
        k = value.decode('utf-8')
        return Get(k)

    def klv(self):
        if False:
            return 10
        v = bytearray(self.key.encode('ascii'))
        action = _klv_utils.get_action_t(Actions.Get, v)
        return bytes(action)

    def __call__(self, store):
        if False:
            for i in range(10):
                print('nop')
        return Get.GetResult(store.storage[self.key])

    def decode_result(self, data):
        if False:
            while True:
                i = 10
        return Get.GetResult.from_klv(data)

class _BarrierImpl:

    def __init__(self, world_size):
        if False:
            print('Hello World!')
        self._world_size = world_size
        self._cvar = threading.Condition()

    def __call__(self):
        if False:
            while True:
                i = 10
        with self._cvar:
            self._world_size -= 1
            if self._world_size == 0:
                self._cvar.notifyAll()
            elif self._world_size > 0:
                self._cvar.wait()

class Barrier:

    class BarrierResult:

        def klv(self):
            if False:
                print('Hello World!')
            v = bytearray(bytes(True))
            action = _klv_utils.get_result_action_t(0, v)
            return bytes(action)

        @staticmethod
        def from_klv(klv):
            if False:
                while True:
                    i = 10
            return True

    def klv(self):
        if False:
            while True:
                i = 10
        action = _klv_utils.get_action_t(Actions.Barrier, bytes(0))
        return bytes(action)

    @staticmethod
    def from_klv(klv):
        if False:
            for i in range(10):
                print('nop')
        return Barrier()

    def __call__(self, store):
        if False:
            while True:
                i = 10
        with store._lock:
            if store._current_barrier is None:
                store._current_barrier = _BarrierImpl(store._world_size)
        store._current_barrier()
        with store._lock:
            store._current_barrier = None
        return Barrier.BarrierResult()

    def decode_result(self, data):
        if False:
            for i in range(10):
                print('nop')
        return Barrier.BarrierResult.from_klv(data)