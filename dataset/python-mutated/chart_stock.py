from . import chart

class ChartStock(chart.Chart):
    """
    A class for writing the Excel XLSX Stock charts.

    """

    def __init__(self, options=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Constructor.\n\n        '
        super(ChartStock, self).__init__()
        self.show_crosses = 0
        self.hi_low_lines = {}
        self.date_category = True
        self.x_axis['defaults']['num_format'] = 'dd/mm/yyyy'
        self.x2_axis['defaults']['num_format'] = 'dd/mm/yyyy'
        self.label_position_default = 'right'
        self.label_positions = {'center': 'ctr', 'right': 'r', 'left': 'l', 'above': 't', 'below': 'b', 'top': 't', 'bottom': 'b'}
        self.set_x_axis({})
        self.set_x2_axis({})

    def _write_chart_type(self, args):
        if False:
            print('Hello World!')
        self._write_stock_chart(args)

    def _write_stock_chart(self, args):
        if False:
            print('Hello World!')
        if args['primary_axes']:
            series = self._get_primary_axes_series()
        else:
            series = self._get_secondary_axes_series()
        if not len(series):
            return
        self._modify_series_formatting()
        self._xml_start_tag('c:stockChart')
        for data in series:
            self._write_ser(data)
        self._write_drop_lines()
        if args.get('primary_axes'):
            self._write_hi_low_lines()
        self._write_up_down_bars()
        self._write_axis_ids(args)
        self._xml_end_tag('c:stockChart')

    def _modify_series_formatting(self):
        if False:
            i = 10
            return i + 15
        index = 0
        for series in self.series:
            if index % 4 != 3:
                if not series['line']['defined']:
                    series['line'] = {'width': 2.25, 'none': 1, 'defined': 1}
                if series['marker'] is None:
                    if index % 4 == 2:
                        series['marker'] = {'type': 'dot', 'size': 3}
                    else:
                        series['marker'] = {'type': 'none'}
            index += 1