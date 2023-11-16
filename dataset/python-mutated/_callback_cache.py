class _PerspectiveCallBackCache(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._callbacks = []

    def add_callback(self, callback):
        if False:
            while True:
                i = 10
        self._callbacks.append(callback)

    def remove_callbacks(self, condition):
        if False:
            i = 10
            return i + 15
        'Remove callback functions that satisfy the given condition.\n\n        Args:\n            condition (func): a function that returns either True or False. If\n                True is returned, filter the item out.\n        '
        if not callable(condition):
            raise ValueError('callback filter condition must be a callable function!')
        self._callbacks = [callback for callback in self._callbacks if condition(callback) is False]

    def pop_callbacks(self, client_id, callback_id):
        if False:
            for i in range(10):
                print('nop')
        'Removes and returns a list of callbacks with the given\n        `callback_id`.\n\n        Args:\n            callback_id (:obj:`str`) an id that identifies the callback.\n\n        Returns:\n            :obj:`list` a list of dicts containing the callbacks that were\n                removed.\n        '
        popped = []
        new_callbacks = []
        for callback in self._callbacks:
            if callback['callback_id'] == callback_id and callback['client_id'] == client_id:
                popped.append(callback)
            else:
                new_callbacks.append(callback)
        self._callbacks = new_callbacks
        return popped

    def get_callbacks(self):
        if False:
            while True:
                i = 10
        return self._callbacks

    def __iter__(self):
        if False:
            while True:
                i = 10
        return iter(self._callbacks)

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self._callbacks)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return str(self._callbacks)