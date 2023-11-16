from . import chart
from warnings import warn

class ChartScatter(chart.Chart):
    """
    A class for writing the Excel XLSX Scatter charts.


    """

    def __init__(self, options=None):
        if False:
            i = 10
            return i + 15
        '\n        Constructor.\n\n        '
        super(ChartScatter, self).__init__()
        if options is None:
            options = {}
        self.subtype = options.get('subtype')
        if not self.subtype:
            self.subtype = 'marker_only'
        self.cross_between = 'midCat'
        self.horiz_val_axis = 0
        self.val_axis_position = 'b'
        self.smooth_allowed = True
        self.requires_category = True
        self.label_position_default = 'right'
        self.label_positions = {'center': 'ctr', 'right': 'r', 'left': 'l', 'above': 't', 'below': 'b', 'top': 't', 'bottom': 'b'}

    def combine(self, chart=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a combination chart with a secondary chart.\n\n        Note: Override parent method to add a warning.\n\n        Args:\n            chart: The secondary chart to combine with the primary chart.\n\n        Returns:\n            Nothing.\n\n        '
        if chart is None:
            return
        warn('Combined chart not currently supported with scatter chart as the primary chart')

    def _write_chart_type(self, args):
        if False:
            for i in range(10):
                print('nop')
        self._write_scatter_chart(args)

    def _write_scatter_chart(self, args):
        if False:
            for i in range(10):
                print('nop')
        if args['primary_axes']:
            series = self._get_primary_axes_series()
        else:
            series = self._get_secondary_axes_series()
        if not len(series):
            return
        style = 'lineMarker'
        subtype = self.subtype
        if subtype == 'marker_only':
            style = 'lineMarker'
        if subtype == 'straight_with_markers':
            style = 'lineMarker'
        if subtype == 'straight':
            style = 'lineMarker'
            self.default_marker = {'type': 'none'}
        if subtype == 'smooth_with_markers':
            style = 'smoothMarker'
        if subtype == 'smooth':
            style = 'smoothMarker'
            self.default_marker = {'type': 'none'}
        self._modify_series_formatting()
        self._xml_start_tag('c:scatterChart')
        self._write_scatter_style(style)
        for data in series:
            self._write_ser(data)
        self._write_axis_ids(args)
        self._xml_end_tag('c:scatterChart')

    def _write_ser(self, series):
        if False:
            i = 10
            return i + 15
        index = self.series_index
        self.series_index += 1
        self._xml_start_tag('c:ser')
        self._write_idx(index)
        self._write_order(index)
        self._write_series_name(series)
        self._write_sp_pr(series)
        self._write_marker(series.get('marker'))
        self._write_d_pt(series.get('points'))
        self._write_d_lbls(series.get('labels'))
        self._write_trendline(series.get('trendline'))
        self._write_error_bars(series.get('error_bars'))
        self._write_x_val(series)
        self._write_y_val(series)
        if 'smooth' in self.subtype and series['smooth'] is None:
            self._write_c_smooth(True)
        else:
            self._write_c_smooth(series['smooth'])
        self._xml_end_tag('c:ser')

    def _write_plot_area(self):
        if False:
            i = 10
            return i + 15
        self._xml_start_tag('c:plotArea')
        self._write_layout(self.plotarea.get('layout'), 'plot')
        self._write_chart_type({'primary_axes': 1})
        self._write_chart_type({'primary_axes': 0})
        self._write_cat_val_axis({'x_axis': self.x_axis, 'y_axis': self.y_axis, 'axis_ids': self.axis_ids, 'position': 'b'})
        tmp = self.horiz_val_axis
        self.horiz_val_axis = 1
        self._write_val_axis({'x_axis': self.x_axis, 'y_axis': self.y_axis, 'axis_ids': self.axis_ids, 'position': 'l'})
        self.horiz_val_axis = tmp
        self._write_cat_val_axis({'x_axis': self.x2_axis, 'y_axis': self.y2_axis, 'axis_ids': self.axis2_ids, 'position': 'b'})
        self.horiz_val_axis = 1
        self._write_val_axis({'x_axis': self.x2_axis, 'y_axis': self.y2_axis, 'axis_ids': self.axis2_ids, 'position': 'l'})
        self._write_sp_pr(self.plotarea)
        self._xml_end_tag('c:plotArea')

    def _write_x_val(self, series):
        if False:
            return 10
        formula = series.get('categories')
        data_id = series.get('cat_data_id')
        data = self.formula_data[data_id]
        self._xml_start_tag('c:xVal')
        data_type = self._get_data_type(data)
        if data_type == 'str':
            self._write_str_ref(formula, data, data_type)
        else:
            self._write_num_ref(formula, data, data_type)
        self._xml_end_tag('c:xVal')

    def _write_y_val(self, series):
        if False:
            print('Hello World!')
        formula = series.get('values')
        data_id = series.get('val_data_id')
        data = self.formula_data[data_id]
        self._xml_start_tag('c:yVal')
        self._write_num_ref(formula, data, 'num')
        self._xml_end_tag('c:yVal')

    def _write_scatter_style(self, val):
        if False:
            print('Hello World!')
        attributes = [('val', val)]
        self._xml_empty_tag('c:scatterStyle', attributes)

    def _modify_series_formatting(self):
        if False:
            while True:
                i = 10
        subtype = self.subtype
        if subtype == 'marker_only':
            for series in self.series:
                if not series['line']['defined']:
                    series['line'] = {'width': 2.25, 'none': 1, 'defined': 1}

    def _write_d_pt_point(self, index, point):
        if False:
            while True:
                i = 10
        self._xml_start_tag('c:dPt')
        self._write_idx(index)
        self._xml_start_tag('c:marker')
        self._write_sp_pr(point)
        self._xml_end_tag('c:marker')
        self._xml_end_tag('c:dPt')