from pgcli.pyev import Visualizer
import json
'Explain response output adapter'

class ExplainOutputFormatter:

    def __init__(self, max_width):
        if False:
            i = 10
            return i + 15
        self.max_width = max_width

    def format_output(self, cur, headers, **output_kwargs):
        if False:
            for i in range(10):
                print('nop')
        [(data,)] = list(cur)
        explain_list = json.loads(data)
        visualizer = Visualizer(self.max_width)
        for explain in explain_list:
            visualizer.load(explain)
            yield visualizer.get_list()