from . import chart
from warnings import warn

class ChartBar(chart.Chart):
    """
    A class for writing the Excel XLSX Bar charts.


    """

    def __init__(self, options=None):
        if False:
            print('Hello World!')
        '\n        Constructor.\n\n        '
        super(ChartBar, self).__init__()
        if options is None:
            options = {}
        self.subtype = options.get('subtype')
        if not self.subtype:
            self.subtype = 'clustered'
        self.cat_axis_position = 'l'
        self.val_axis_position = 'b'
        self.horiz_val_axis = 0
        self.horiz_cat_axis = 1
        self.show_crosses = 0
        self.x_axis['defaults']['major_gridlines'] = {'visible': 1}
        self.y_axis['defaults']['major_gridlines'] = {'visible': 0}
        if self.subtype == 'percent_stacked':
            self.x_axis['defaults']['num_format'] = '0%'
        self.label_position_default = 'outside_end'
        self.label_positions = {'center': 'ctr', 'inside_base': 'inBase', 'inside_end': 'inEnd', 'outside_end': 'outEnd'}
        self.set_x_axis({})
        self.set_y_axis({})

    def combine(self, chart=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a combination chart with a secondary chart.\n\n        Note: Override parent method to add an extra check that is required\n        for Bar charts to ensure that their combined chart is on a secondary\n        axis.\n\n        Args:\n            chart: The secondary chart to combine with the primary chart.\n\n        Returns:\n            Nothing.\n\n        '
        if chart is None:
            return
        if not chart.is_secondary:
            warn('Charts combined with Bar charts must be on a secondary axis')
        self.combined = chart

    def _write_chart_type(self, args):
        if False:
            print('Hello World!')
        if args['primary_axes']:
            tmp = self.y_axis
            self.y_axis = self.x_axis
            self.x_axis = tmp
            if self.y2_axis['position'] == 'r':
                self.y2_axis['position'] = 't'
        self._write_bar_chart(args)

    def _write_bar_chart(self, args):
        if False:
            i = 10
            return i + 15
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
            while True:
                i = 10
        val = 'bar'
        attributes = [('val', val)]
        self._xml_empty_tag('c:barDir', attributes)

    def _write_err_dir(self, val):
        if False:
            print('Hello World!')
        pass