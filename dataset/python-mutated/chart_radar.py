from . import chart

class ChartRadar(chart.Chart):
    """
    A class for writing the Excel XLSX Radar charts.


    """

    def __init__(self, options=None):
        if False:
            return 10
        '\n        Constructor.\n\n        '
        super(ChartRadar, self).__init__()
        if options is None:
            options = {}
        self.subtype = options.get('subtype')
        if not self.subtype:
            self.subtype = 'marker'
            self.default_marker = {'type': 'none'}
        self.x_axis['defaults']['major_gridlines'] = {'visible': 1}
        self.set_x_axis({})
        self.label_position_default = 'center'
        self.label_positions = {'center': 'ctr'}
        self.y_axis['major_tick_mark'] = 'cross'

    def _write_chart_type(self, args):
        if False:
            return 10
        self._write_radar_chart(args)

    def _write_radar_chart(self, args):
        if False:
            i = 10
            return i + 15
        if args['primary_axes']:
            series = self._get_primary_axes_series()
        else:
            series = self._get_secondary_axes_series()
        if not len(series):
            return
        self._xml_start_tag('c:radarChart')
        self._write_radar_style()
        for data in series:
            self._write_ser(data)
        self._write_axis_ids(args)
        self._xml_end_tag('c:radarChart')

    def _write_radar_style(self):
        if False:
            while True:
                i = 10
        val = 'marker'
        if self.subtype == 'filled':
            val = 'filled'
        attributes = [('val', val)]
        self._xml_empty_tag('c:radarStyle', attributes)