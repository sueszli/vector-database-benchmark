import pandas as pd
from qlib.data.inst_processor import InstProcessor
from qlib.utils.resam import resam_calendar

class ResampleNProcessor(InstProcessor):

    def __init__(self, target_frq: str, **kwargs):
        if False:
            i = 10
            return i + 15
        self.target_frq = target_frq

    def __call__(self, df: pd.DataFrame, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        df.index = pd.to_datetime(df.index)
        res_index = resam_calendar(df.index, '1min', self.target_frq)
        df = df.resample(self.target_frq).last().reindex(res_index)
        return df