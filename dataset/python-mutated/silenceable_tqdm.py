import os
import tqdm

class SilenceableTqdm(tqdm.tqdm):
    """
    Wrapper for tqdm that disables all progress bars if HAYSTACK_PROGRESS_BARS is set to a falsey value
    ("0", "False", "FALSE", "false").

    Note: this check is done every time a tqdm iterator is initialized, so normally for each method run. Therefore
    progress bars can be enabled and disabled at runtime, but not during a specific iteration.
    """

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        '\n        Passes `disable=True` to tqdm if `self.no_progress_bars` is set to True.\n        '
        if self.no_progress_bars:
            kwargs['disable'] = True
        super().__init__(*args, **kwargs)

    @property
    def no_progress_bars(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Reads the HAYSTACK_PROGRESS_BARS env var to check if the progress bars should be disabled.\n        '
        return os.getenv('HAYSTACK_PROGRESS_BARS', '1') in ['0', 'False', 'FALSE', 'false']

    @property
    def disable(self):
        if False:
            print('Hello World!')
        return self.no_progress_bars or self._disable

    @disable.setter
    def disable(self, value):
        if False:
            i = 10
            return i + 15
        self._disable = value
tqdm.std.tqdm = SilenceableTqdm
tqdm.tqdm = SilenceableTqdm