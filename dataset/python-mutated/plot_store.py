from collections import defaultdict
from rqalpha.environment import Environment
from rqalpha.core.execution_context import ExecutionContext
from rqalpha.utils.arg_checker import apply_rules, verify_that
from rqalpha.const import EXECUTION_PHASE

class PlotStore(object):

    def __init__(self, env):
        if False:
            print('Hello World!')
        self._env = env
        self._plots = defaultdict(dict)

    def add_plot(self, dt, series_name, value):
        if False:
            return 10
        self._plots[series_name][dt] = value

    def get_plots(self):
        if False:
            for i in range(10):
                print('nop')
        return self._plots

    @ExecutionContext.enforce_phase(EXECUTION_PHASE.ON_BAR, EXECUTION_PHASE.ON_TICK, EXECUTION_PHASE.SCHEDULED)
    @apply_rules(verify_that('series_name', pre_check=True).is_instance_of(str), verify_that('value', pre_check=True).is_number())
    def plot(self, series_name, value):
        if False:
            while True:
                i = 10
        '\n        在策略运行结束后的收益图中，加入自定义的曲线。\n        每次调用 plot 函数将会以当前时间为横坐标，value 为纵坐标加入一个点，series_name 相同的点将连成一条曲线。\n\n        :param series_name: 曲线名称\n        :param value: 点的纵坐标值\n\n        :example:\n\n        .. code-block:: python\n\n            def handle_bar(context, bar_dict):\n                plot("OPEN", bar_dict["000001.XSHE"].open)\n\n        '
        self.add_plot(self._env.trading_dt.date(), series_name, value)