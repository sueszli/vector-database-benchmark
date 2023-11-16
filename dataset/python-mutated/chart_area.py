from . import chart

class ChartArea(chart.Chart):
    """
    A class for writing the Excel XLSX Area charts.


    """

    def __init__(self, options=None):
        if False:
            while True:
                i = 10
        '\n        Constructor.\n\n        '
        super(ChartArea, self).__init__()
        if options is None:
            options = {}
        self.subtype = options.get('subtype')
        if not self.subtype:
            self.subtype = 'standard'
        self.cross_between = 'midCat'
        self.show_crosses = 0
        if self.subtype == 'percent_stacked':
            self.y_axis['defaults']['num_format'] = '0%'
        self.label_position_default = 'center'
        self.label_positions = {'center': 'ctr'}
        self.set_y_axis({})

    def _write_chart_type(self, args):
        if False:
            print('Hello World!')
        self._write_area_chart(args)

    def _write_area_chart(self, args):
        if False:
            print('Hello World!')
        if args['primary_axes']:
            series = self._get_primary_axes_series()
        else:
            series = self._get_secondary_axes_series()
        if not len(series):
            return
        subtype = self.subtype
        if subtype == 'percent_stacked':
            subtype = 'percentStacked'
        self._xml_start_tag('c:areaChart')
        self._write_grouping(subtype)
        for data in series:
            self._write_ser(data)
        self._write_drop_lines()
        self._write_axis_ids(args)
        self._xml_end_tag('c:areaChart')