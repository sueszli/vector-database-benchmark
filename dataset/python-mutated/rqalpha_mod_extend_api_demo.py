import os
import pandas as pd
from rqalpha.interface import AbstractMod
__config__ = {'csv_path': None}

def load_mod():
    if False:
        print('Hello World!')
    return ExtendAPIDemoMod()

class ExtendAPIDemoMod(AbstractMod):

    def __init__(self):
        if False:
            while True:
                i = 10
        self._csv_path = None
        self._inject_api()

    def start_up(self, env, mod_config):
        if False:
            print('Hello World!')
        self._csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), mod_config.csv_path))

    def tear_down(self, code, exception=None):
        if False:
            return 10
        pass

    def _inject_api(self):
        if False:
            i = 10
            return i + 15
        from rqalpha.api import export_as_api
        from rqalpha.core.execution_context import ExecutionContext
        from rqalpha.const import EXECUTION_PHASE

        @export_as_api
        @ExecutionContext.enforce_phase(EXECUTION_PHASE.ON_INIT, EXECUTION_PHASE.BEFORE_TRADING, EXECUTION_PHASE.ON_BAR, EXECUTION_PHASE.AFTER_TRADING, EXECUTION_PHASE.SCHEDULED)
        def get_csv_as_df():
            if False:
                while True:
                    i = 10
            data = pd.read_csv(self._csv_path)
            return data