from . import chart

class ChartLine(chart.Chart):
    """
    A class for writing the Excel XLSX Line charts.


    """

    def __init__(self, options=None):
        if False:
            print('Hello World!')
        '\n        Constructor.\n\n        '
        super(ChartLine, self).__init__()
        if options is None:
            options = {}
        self.subtype = options.get('subtype')
        if not self.subtype:
            self.subtype = 'standard'
        self.default_marker = {'type': 'none'}
        self.smooth_allowed = True
        if self.subtype == 'percent_stacked':
            self.y_axis['defaults']['num_format'] = '0%'
        self.label_position_default = 'right'
        self.label_positions = {'center': 'ctr', 'right': 'r', 'left': 'l', 'above': 't', 'below': 'b', 'top': 't', 'bottom': 'b'}
        self.set_y_axis({})

    def _write_chart_type(self, args):
        if False:
            return 10
        self._write_line_chart(args)

    def _write_line_chart(self, args):
        if False:
            for i in range(10):
                print('nop')
        if args['primary_axes']:
            series = self._get_primary_axes_series()
        else:
            series = self._get_secondary_axes_series()
        if not len(series):
            return
        subtype = self.subtype
        if subtype == 'percent_stacked':
            subtype = 'percentStacked'
        self._xml_start_tag('c:lineChart')
        self._write_grouping(subtype)
        for data in series:
            self._write_ser(data)
        self._write_drop_lines()
        self._write_hi_low_lines()
        self._write_up_down_bars()
        self._write_marker_value()
        self._write_axis_ids(args)
        self._xml_end_tag('c:lineChart')

    def _write_d_pt_point(self, index, point):
        if False:
            return 10
        self._xml_start_tag('c:dPt')
        self._write_idx(index)
        self._xml_start_tag('c:marker')
        self._write_sp_pr(point)
        self._xml_end_tag('c:marker')
        self._xml_end_tag('c:dPt')

    def _write_marker_value(self):
        if False:
            for i in range(10):
                print('nop')
        attributes = [('val', 1)]
        self._xml_empty_tag('c:marker', attributes)