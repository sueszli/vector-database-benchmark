import tree
from keras.api_export import keras_export
from keras.callbacks.callback import Callback
from keras.callbacks.history import History
from keras.callbacks.progbar_logger import ProgbarLogger

@keras_export('keras.callbacks.CallbackList')
class CallbackList(Callback):
    """Container abstracting a list of callbacks."""

    def __init__(self, callbacks=None, add_history=False, add_progbar=False, model=None, **params):
        if False:
            while True:
                i = 10
        'Container for `Callback` instances.\n\n        This object wraps a list of `Callback` instances, making it possible\n        to call them all at once via a single endpoint\n        (e.g. `callback_list.on_epoch_end(...)`).\n\n        Args:\n            callbacks: List of `Callback` instances.\n            add_history: Whether a `History` callback should be added, if one\n                does not already exist in the `callbacks` list.\n            add_progbar: Whether a `ProgbarLogger` callback should be added, if\n                one does not already exist in the `callbacks` list.\n            model: The `Model` these callbacks are used with.\n            **params: If provided, parameters will be passed to each `Callback`\n                via `Callback.set_params`.\n        '
        self.callbacks = tree.flatten(callbacks) if callbacks else []
        self._add_default_callbacks(add_history, add_progbar)
        if model:
            self.set_model(model)
        if params:
            self.set_params(params)

    def _add_default_callbacks(self, add_history, add_progbar):
        if False:
            return 10
        'Adds `Callback`s that are always present.'
        self._progbar = None
        self._history = None
        for cb in self.callbacks:
            if isinstance(cb, ProgbarLogger):
                self._progbar = cb
            elif isinstance(cb, History):
                self._history = cb
        if self._history is None and add_history:
            self._history = History()
            self.callbacks.append(self._history)
        if self._progbar is None and add_progbar:
            self._progbar = ProgbarLogger()
            self.callbacks.append(self._progbar)

    def append(self, callback):
        if False:
            while True:
                i = 10
        self.callbacks.append(callback)

    def set_params(self, params):
        if False:
            for i in range(10):
                print('nop')
        self.params = params
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        if False:
            print('Hello World!')
        super().set_model(model)
        if self._history:
            model.history = self._history
        for callback in self.callbacks:
            callback.set_model(model)

    def on_batch_begin(self, batch, logs=None):
        if False:
            print('Hello World!')
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        if False:
            return 10
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs=logs)

    def on_epoch_begin(self, epoch, logs=None):
        if False:
            for i in range(10):
                print('nop')
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        if False:
            for i in range(10):
                print('nop')
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_train_batch_begin(self, batch, logs=None):
        if False:
            for i in range(10):
                print('nop')
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_batch_begin(batch, logs=logs)

    def on_train_batch_end(self, batch, logs=None):
        if False:
            i = 10
            return i + 15
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_batch_end(batch, logs=logs)

    def on_test_batch_begin(self, batch, logs=None):
        if False:
            return 10
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_test_batch_begin(batch, logs=logs)

    def on_test_batch_end(self, batch, logs=None):
        if False:
            while True:
                i = 10
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_test_batch_end(batch, logs=logs)

    def on_predict_batch_begin(self, batch, logs=None):
        if False:
            while True:
                i = 10
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_predict_batch_begin(batch, logs=logs)

    def on_predict_batch_end(self, batch, logs=None):
        if False:
            i = 10
            return i + 15
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_predict_batch_end(batch, logs=logs)

    def on_train_begin(self, logs=None):
        if False:
            for i in range(10):
                print('nop')
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        if False:
            while True:
                i = 10
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_test_begin(self, logs=None):
        if False:
            print('Hello World!')
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_test_begin(logs)

    def on_test_end(self, logs=None):
        if False:
            return 10
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_test_end(logs)

    def on_predict_begin(self, logs=None):
        if False:
            print('Hello World!')
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_predict_begin(logs)

    def on_predict_end(self, logs=None):
        if False:
            while True:
                i = 10
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_predict_end(logs)