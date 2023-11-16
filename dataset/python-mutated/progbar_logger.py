from keras.api_export import keras_export
from keras.callbacks.callback import Callback
from keras.utils import io_utils
from keras.utils.progbar import Progbar

@keras_export('keras.callbacks.ProgbarLogger')
class ProgbarLogger(Callback):
    """Callback that prints metrics to stdout.

    Args:
        count_mode: One of `"steps"` or `"samples"`.
            Whether the progress bar should
            count samples seen or steps (batches) seen.

    Raises:
        ValueError: In case of invalid `count_mode`.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.seen = 0
        self.progbar = None
        self.target = None
        self.verbose = 1
        self.epochs = 1
        self._called_in_fit = False

    def set_params(self, params):
        if False:
            i = 10
            return i + 15
        verbose = params['verbose']
        if verbose == 'auto':
            verbose = 1
        self.verbose = verbose
        self.epochs = params['epochs']
        self.target = params['steps']

    def on_train_begin(self, logs=None):
        if False:
            return 10
        self._called_in_fit = True

    def on_test_begin(self, logs=None):
        if False:
            print('Hello World!')
        if not self._called_in_fit:
            self._reset_progbar()
            self._maybe_init_progbar()

    def on_predict_begin(self, logs=None):
        if False:
            i = 10
            return i + 15
        self._reset_progbar()
        self._maybe_init_progbar()

    def on_epoch_begin(self, epoch, logs=None):
        if False:
            while True:
                i = 10
        self._reset_progbar()
        self._maybe_init_progbar()
        if self.verbose and self.epochs > 1:
            io_utils.print_msg(f'Epoch {epoch + 1}/{self.epochs}')

    def on_train_batch_end(self, batch, logs=None):
        if False:
            return 10
        self._update_progbar(batch, logs)

    def on_test_batch_end(self, batch, logs=None):
        if False:
            while True:
                i = 10
        if not self._called_in_fit:
            self._update_progbar(batch, logs)

    def on_predict_batch_end(self, batch, logs=None):
        if False:
            for i in range(10):
                print('nop')
        self._update_progbar(batch, None)

    def on_epoch_end(self, epoch, logs=None):
        if False:
            for i in range(10):
                print('nop')
        self._finalize_progbar(logs)

    def on_test_end(self, logs=None):
        if False:
            return 10
        if not self._called_in_fit:
            self._finalize_progbar(logs)

    def on_predict_end(self, logs=None):
        if False:
            while True:
                i = 10
        self._finalize_progbar(logs)

    def _reset_progbar(self):
        if False:
            return 10
        self.seen = 0
        self.progbar = None

    def _maybe_init_progbar(self):
        if False:
            return 10
        if self.progbar is None:
            self.progbar = Progbar(target=self.target, verbose=self.verbose, unit_name='step')

    def _update_progbar(self, batch, logs=None):
        if False:
            return 10
        'Updates the progbar.'
        logs = logs or {}
        self._maybe_init_progbar()
        self.seen = batch + 1
        if self.verbose == 1:
            self.progbar.update(self.seen, list(logs.items()), finalize=False)

    def _finalize_progbar(self, logs):
        if False:
            while True:
                i = 10
        logs = logs or {}
        if self.target is None:
            self.target = self.seen
            self.progbar.target = self.target
        self.progbar.update(self.target, list(logs.items()), finalize=True)