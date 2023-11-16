from . import chart

class ChartColumn(chart.Chart):
    """
    A class for writing the Excel XLSX Column charts.


    """

    def __init__(self, options=None):
        if False:
            print('Hello World!')
        '\n        Constructor.\n\n        '
        super(ChartColumn, self).__init__()
        if options is None:
            options = {}
        self.subtype = options.get('subtype')
        if not self.subtype:
            self.subtype = 'clustered'
        self.horiz_val_axis = 0
        if self.subtype == 'percent_stacked':
            self.y_axis['defaults']['num_format'] = '0%'
        self.label_position_default = 'outside_end'
        self.label_positions = {'center': 'ctr', 'inside_base': 'inBase', 'inside_end': 'inEnd', 'outside_end': 'outEnd'}
        self.set_y_axis({})

    def _write_chart_type(self, args):
        if False:
            for i in range(10):
                print('nop')
        self._write_bar_chart(args)

    def _write_bar_chart(self, args):
        if False:
            return 10
        if args['primary_axes']:
            series = self._get_primary_axes_series()
        else:
            series = self._get_secondary_axes_series()
        if not len(series):
            return
        subtype = self.subtype
        if subtype == 'percent_stacked':
            subtype = 'percentStacked'
        if 'stacked' in self.subtype and self.series_overlap_1 is None:
            self.series_overlap_1 = 100
        self._xml_start_tag('c:barChart')
        self._write_bar_dir()
        self._write_grouping(subtype)
        for data in series:
            self._write_ser(data)
        if args['primary_axes']:
            self._write_gap_width(self.series_gap_1)
        else:
            self._write_gap_width(self.series_gap_2)
        if args['primary_axes']:
            self._write_overlap(self.series_overlap_1)
        else:
            self._write_overlap(self.series_overlap_2)
        self._write_axis_ids(args)
        self._xml_end_tag('c:barChart')

    def _write_bar_dir(self):
        if False:
            return 10
        val = 'col'
        attributes = [('val', val)]
        self._xml_empty_tag('c:barDir', attributes)

    def _write_err_dir(self, val):
        if False:
            print('Hello World!')
        pass