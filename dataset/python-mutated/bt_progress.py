from freqtrade.enums import BacktestState

class BTProgress:
    _action: BacktestState = BacktestState.STARTUP
    _progress: float = 0
    _max_steps: float = 0

    def __init__(self):
        if False:
            return 10
        pass

    def init_step(self, action: BacktestState, max_steps: float):
        if False:
            for i in range(10):
                print('nop')
        self._action = action
        self._max_steps = max_steps
        self._progress = 0

    def set_new_value(self, new_value: float):
        if False:
            return 10
        self._progress = new_value

    def increment(self):
        if False:
            for i in range(10):
                print('nop')
        self._progress += 1

    @property
    def progress(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get progress as ratio, capped to be between 0 and 1 (to avoid small calculation errors).\n        '
        return max(min(round(self._progress / self._max_steps, 5) if self._max_steps > 0 else 0, 1), 0)

    @property
    def action(self):
        if False:
            while True:
                i = 10
        return str(self._action)