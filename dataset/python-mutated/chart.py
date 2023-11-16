import re
import copy
from warnings import warn
from .shape import Shape
from . import xmlwriter
from .utility import get_rgb_color
from .utility import xl_rowcol_to_cell
from .utility import xl_range_formula
from .utility import supported_datetime
from .utility import datetime_to_excel_datetime
from .utility import quote_sheetname

class Chart(xmlwriter.XMLwriter):
    """
    A class for writing the Excel XLSX Chart file.


    """

    def __init__(self, options=None):
        if False:
            i = 10
            return i + 15
        '\n        Constructor.\n\n        '
        super(Chart, self).__init__()
        self.subtype = None
        self.sheet_type = 512
        self.orientation = 0
        self.series = []
        self.embedded = 0
        self.id = -1
        self.series_index = 0
        self.style_id = 2
        self.axis_ids = []
        self.axis2_ids = []
        self.cat_has_num_fmt = 0
        self.requires_category = False
        self.legend = {}
        self.cat_axis_position = 'b'
        self.val_axis_position = 'l'
        self.formula_ids = {}
        self.formula_data = []
        self.horiz_cat_axis = 0
        self.horiz_val_axis = 1
        self.protection = 0
        self.chartarea = {}
        self.plotarea = {}
        self.x_axis = {}
        self.y_axis = {}
        self.y2_axis = {}
        self.x2_axis = {}
        self.chart_name = ''
        self.show_blanks = 'gap'
        self.show_na_as_empty = False
        self.show_hidden = False
        self.show_crosses = 1
        self.width = 480
        self.height = 288
        self.x_scale = 1
        self.y_scale = 1
        self.x_offset = 0
        self.y_offset = 0
        self.table = None
        self.cross_between = 'between'
        self.default_marker = None
        self.series_gap_1 = None
        self.series_gap_2 = None
        self.series_overlap_1 = None
        self.series_overlap_2 = None
        self.drop_lines = None
        self.hi_low_lines = None
        self.up_down_bars = None
        self.smooth_allowed = False
        self.title_font = None
        self.title_name = None
        self.title_formula = None
        self.title_data_id = None
        self.title_layout = None
        self.title_overlay = None
        self.title_none = False
        self.date_category = False
        self.date_1904 = False
        self.remove_timezone = False
        self.label_positions = {}
        self.label_position_default = ''
        self.already_inserted = False
        self.combined = None
        self.is_secondary = False
        self.warn_sheetname = True
        self._set_default_properties()

    def add_series(self, options=None):
        if False:
            while True:
                i = 10
        '\n        Add a data series to a chart.\n\n        Args:\n            options:  A dictionary of chart series options.\n\n        Returns:\n            Nothing.\n\n        '
        if options is None:
            options = {}
        if 'values' not in options:
            warn("Must specify 'values' in add_series()")
            return
        if self.requires_category and 'categories' not in options:
            warn("Must specify 'categories' in add_series() for this chart type")
            return
        if len(self.series) == 255:
            warn('The maximum number of series that can be added to an Excel Chart is 255')
            return
        values = self._list_to_formula(options.get('values'))
        categories = self._list_to_formula(options.get('categories'))
        (name, name_formula) = self._process_names(options.get('name'), options.get('name_formula'))
        cat_id = self._get_data_id(categories, options.get('categories_data'))
        val_id = self._get_data_id(values, options.get('values_data'))
        name_id = self._get_data_id(name_formula, options.get('name_data'))
        line = Shape._get_line_properties(options.get('line'))
        if options.get('border'):
            line = Shape._get_line_properties(options['border'])
        fill = Shape._get_fill_properties(options.get('fill'))
        pattern = Shape._get_pattern_properties(options.get('pattern'))
        gradient = Shape._get_gradient_properties(options.get('gradient'))
        if pattern:
            self.fill = None
        if gradient:
            pattern = None
            fill = None
        marker = self._get_marker_properties(options.get('marker'))
        trendline = self._get_trendline_properties(options.get('trendline'))
        smooth = options.get('smooth')
        y_error_bars = self._get_error_bars_props(options.get('y_error_bars'))
        x_error_bars = self._get_error_bars_props(options.get('x_error_bars'))
        error_bars = {'x_error_bars': x_error_bars, 'y_error_bars': y_error_bars}
        points = self._get_points_properties(options.get('points'))
        labels = self._get_labels_properties(options.get('data_labels'))
        invert_if_neg = options.get('invert_if_negative', False)
        inverted_color = options.get('invert_if_negative_color', False)
        x2_axis = options.get('x2_axis')
        y2_axis = options.get('y2_axis')
        if x2_axis or y2_axis:
            self.is_secondary = True
        if options.get('gap') is not None:
            if y2_axis:
                self.series_gap_2 = options['gap']
            else:
                self.series_gap_1 = options['gap']
        if options.get('overlap'):
            if y2_axis:
                self.series_overlap_2 = options['overlap']
            else:
                self.series_overlap_1 = options['overlap']
        series = {'values': values, 'categories': categories, 'name': name, 'name_formula': name_formula, 'name_id': name_id, 'val_data_id': val_id, 'cat_data_id': cat_id, 'line': line, 'fill': fill, 'pattern': pattern, 'gradient': gradient, 'marker': marker, 'trendline': trendline, 'labels': labels, 'invert_if_neg': invert_if_neg, 'inverted_color': inverted_color, 'x2_axis': x2_axis, 'y2_axis': y2_axis, 'points': points, 'error_bars': error_bars, 'smooth': smooth}
        self.series.append(series)

    def set_x_axis(self, options):
        if False:
            i = 10
            return i + 15
        '\n        Set the chart X axis options.\n\n        Args:\n            options:  A dictionary of axis options.\n\n        Returns:\n            Nothing.\n\n        '
        axis = self._convert_axis_args(self.x_axis, options)
        self.x_axis = axis

    def set_y_axis(self, options):
        if False:
            return 10
        '\n        Set the chart Y axis options.\n\n        Args:\n            options: A dictionary of axis options.\n\n        Returns:\n            Nothing.\n\n        '
        axis = self._convert_axis_args(self.y_axis, options)
        self.y_axis = axis

    def set_x2_axis(self, options):
        if False:
            i = 10
            return i + 15
        '\n        Set the chart secondary X axis options.\n\n        Args:\n            options: A dictionary of axis options.\n\n        Returns:\n            Nothing.\n\n        '
        axis = self._convert_axis_args(self.x2_axis, options)
        self.x2_axis = axis

    def set_y2_axis(self, options):
        if False:
            while True:
                i = 10
        '\n        Set the chart secondary Y axis options.\n\n        Args:\n            options: A dictionary of axis options.\n\n        Returns:\n            Nothing.\n\n        '
        axis = self._convert_axis_args(self.y2_axis, options)
        self.y2_axis = axis

    def set_title(self, options=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the chart title options.\n\n        Args:\n            options: A dictionary of chart title options.\n\n        Returns:\n            Nothing.\n\n        '
        if options is None:
            options = {}
        (name, name_formula) = self._process_names(options.get('name'), options.get('name_formula'))
        data_id = self._get_data_id(name_formula, options.get('data'))
        self.title_name = name
        self.title_formula = name_formula
        self.title_data_id = data_id
        self.title_font = self._convert_font_args(options.get('name_font'))
        self.title_layout = self._get_layout_properties(options.get('layout'), True)
        self.title_overlay = options.get('overlay')
        self.title_none = options.get('none')

    def set_legend(self, options):
        if False:
            while True:
                i = 10
        '\n        Set the chart legend options.\n\n        Args:\n            options: A dictionary of chart legend options.\n\n        Returns:\n            Nothing.\n        '
        self.legend = self._get_legend_properties(options)

    def set_plotarea(self, options):
        if False:
            print('Hello World!')
        '\n        Set the chart plot area options.\n\n        Args:\n            options: A dictionary of chart plot area options.\n\n        Returns:\n            Nothing.\n        '
        self.plotarea = self._get_area_properties(options)

    def set_chartarea(self, options):
        if False:
            i = 10
            return i + 15
        '\n        Set the chart area options.\n\n        Args:\n            options: A dictionary of chart area options.\n\n        Returns:\n            Nothing.\n        '
        self.chartarea = self._get_area_properties(options)

    def set_style(self, style_id):
        if False:
            print('Hello World!')
        '\n        Set the chart style type.\n\n        Args:\n            style_id: An int representing the chart style.\n\n        Returns:\n            Nothing.\n        '
        if style_id is None:
            style_id = 2
        if style_id < 1 or style_id > 48:
            style_id = 2
        self.style_id = style_id

    def show_blanks_as(self, option):
        if False:
            while True:
                i = 10
        '\n        Set the option for displaying blank data in a chart.\n\n        Args:\n            option: A string representing the display option.\n\n        Returns:\n            Nothing.\n        '
        if not option:
            return
        valid_options = {'gap': 1, 'zero': 1, 'span': 1}
        if option not in valid_options:
            warn("Unknown show_blanks_as() option '%s'" % option)
            return
        self.show_blanks = option

    def show_na_as_empty_cell(self):
        if False:
            print('Hello World!')
        '\n        Display ``#N/A`` on charts as blank/empty cells.\n\n        Args:\n            None.\n\n        Returns:\n            Nothing.\n        '
        self.show_na_as_empty = True

    def show_hidden_data(self):
        if False:
            return 10
        '\n        Display data on charts from hidden rows or columns.\n\n        Args:\n            None.\n\n        Returns:\n            Nothing.\n        '
        self.show_hidden = True

    def set_size(self, options=None):
        if False:
            i = 10
            return i + 15
        '\n        Set size or scale of the chart.\n\n        Args:\n            options: A dictionary of chart size options.\n\n        Returns:\n            Nothing.\n        '
        if options is None:
            options = {}
        self.width = options.get('width', self.width)
        self.height = options.get('height', self.height)
        self.x_scale = options.get('x_scale', 1)
        self.y_scale = options.get('y_scale', 1)
        self.x_offset = options.get('x_offset', 0)
        self.y_offset = options.get('y_offset', 0)

    def set_table(self, options=None):
        if False:
            print('Hello World!')
        '\n        Set properties for an axis data table.\n\n        Args:\n            options: A dictionary of axis table options.\n\n        Returns:\n            Nothing.\n\n        '
        if options is None:
            options = {}
        table = {}
        table['horizontal'] = options.get('horizontal', 1)
        table['vertical'] = options.get('vertical', 1)
        table['outline'] = options.get('outline', 1)
        table['show_keys'] = options.get('show_keys', 0)
        table['font'] = self._convert_font_args(options.get('font'))
        self.table = table

    def set_up_down_bars(self, options=None):
        if False:
            i = 10
            return i + 15
        '\n        Set properties for the chart up-down bars.\n\n        Args:\n            options: A dictionary of options.\n\n        Returns:\n            Nothing.\n\n        '
        if options is None:
            options = {}
        up_line = None
        up_fill = None
        down_line = None
        down_fill = None
        if options.get('up'):
            if 'border' in options['up']:
                up_line = Shape._get_line_properties(options['up']['border'])
            if 'line' in options['up']:
                up_line = Shape._get_line_properties(options['up']['line'])
            if 'fill' in options['up']:
                up_fill = Shape._get_fill_properties(options['up']['fill'])
        if options.get('down'):
            if 'border' in options['down']:
                down_line = Shape._get_line_properties(options['down']['border'])
            if 'line' in options['down']:
                down_line = Shape._get_line_properties(options['down']['line'])
            if 'fill' in options['down']:
                down_fill = Shape._get_fill_properties(options['down']['fill'])
        self.up_down_bars = {'up': {'line': up_line, 'fill': up_fill}, 'down': {'line': down_line, 'fill': down_fill}}

    def set_drop_lines(self, options=None):
        if False:
            while True:
                i = 10
        '\n        Set properties for the chart drop lines.\n\n        Args:\n            options: A dictionary of options.\n\n        Returns:\n            Nothing.\n\n        '
        if options is None:
            options = {}
        line = Shape._get_line_properties(options.get('line'))
        fill = Shape._get_fill_properties(options.get('fill'))
        pattern = Shape._get_pattern_properties(options.get('pattern'))
        gradient = Shape._get_gradient_properties(options.get('gradient'))
        if pattern:
            self.fill = None
        if gradient:
            pattern = None
            fill = None
        self.drop_lines = {'line': line, 'fill': fill, 'pattern': pattern, 'gradient': gradient}

    def set_high_low_lines(self, options=None):
        if False:
            return 10
        '\n        Set properties for the chart high-low lines.\n\n        Args:\n            options: A dictionary of options.\n\n        Returns:\n            Nothing.\n\n        '
        if options is None:
            options = {}
        line = Shape._get_line_properties(options.get('line'))
        fill = Shape._get_fill_properties(options.get('fill'))
        pattern = Shape._get_pattern_properties(options.get('pattern'))
        gradient = Shape._get_gradient_properties(options.get('gradient'))
        if pattern:
            self.fill = None
        if gradient:
            pattern = None
            fill = None
        self.hi_low_lines = {'line': line, 'fill': fill, 'pattern': pattern, 'gradient': gradient}

    def combine(self, chart=None):
        if False:
            while True:
                i = 10
        '\n        Create a combination chart with a secondary chart.\n\n        Args:\n            chart: The secondary chart to combine with the primary chart.\n\n        Returns:\n            Nothing.\n\n        '
        if chart is None:
            return
        self.combined = chart

    def _assemble_xml_file(self):
        if False:
            while True:
                i = 10
        self._xml_declaration()
        self._write_chart_space()
        self._write_lang()
        self._write_style()
        self._write_protection()
        self._write_chart()
        self._write_sp_pr(self.chartarea)
        if self.embedded:
            self._write_print_settings()
        self._xml_end_tag('c:chartSpace')
        self._xml_close()

    def _convert_axis_args(self, axis, user_options):
        if False:
            for i in range(10):
                print('nop')
        options = axis['defaults'].copy()
        options.update(user_options)
        (name, name_formula) = self._process_names(options.get('name'), options.get('name_formula'))
        data_id = self._get_data_id(name_formula, options.get('data'))
        axis = {'defaults': axis['defaults'], 'name': name, 'formula': name_formula, 'data_id': data_id, 'reverse': options.get('reverse'), 'min': options.get('min'), 'max': options.get('max'), 'minor_unit': options.get('minor_unit'), 'major_unit': options.get('major_unit'), 'minor_unit_type': options.get('minor_unit_type'), 'major_unit_type': options.get('major_unit_type'), 'display_units': options.get('display_units'), 'log_base': options.get('log_base'), 'crossing': options.get('crossing'), 'position_axis': options.get('position_axis'), 'position': options.get('position'), 'label_position': options.get('label_position'), 'label_align': options.get('label_align'), 'num_format': options.get('num_format'), 'num_format_linked': options.get('num_format_linked'), 'interval_unit': options.get('interval_unit'), 'interval_tick': options.get('interval_tick'), 'text_axis': False}
        if 'visible' in options:
            axis['visible'] = options.get('visible')
        else:
            axis['visible'] = 1
        axis['display_units'] = self._get_display_units(axis['display_units'])
        axis['display_units_visible'] = options.get('display_units_visible', True)
        if options.get('major_gridlines') and options['major_gridlines']['visible']:
            axis['major_gridlines'] = self._get_gridline_properties(options['major_gridlines'])
        if options.get('minor_gridlines') and options['minor_gridlines']['visible']:
            axis['minor_gridlines'] = self._get_gridline_properties(options['minor_gridlines'])
        if axis.get('position'):
            axis['position'] = axis['position'].lower()[0]
        if axis.get('position_axis'):
            if axis['position_axis'] == 'on_tick':
                axis['position_axis'] = 'midCat'
            elif axis['position_axis'] == 'between':
                pass
            else:
                axis['position_axis'] = None
        if options.get('date_axis'):
            self.date_category = True
        if options.get('text_axis'):
            self.date_category = False
            axis['text_axis'] = True
        if axis.get('min') and supported_datetime(axis['min']):
            axis['min'] = datetime_to_excel_datetime(axis['min'], self.date_1904, self.remove_timezone)
        if axis.get('max') and supported_datetime(axis['max']):
            axis['max'] = datetime_to_excel_datetime(axis['max'], self.date_1904, self.remove_timezone)
        if axis.get('crossing') and supported_datetime(axis['crossing']):
            axis['crossing'] = datetime_to_excel_datetime(axis['crossing'], self.date_1904, self.remove_timezone)
        axis['num_font'] = self._convert_font_args(options.get('num_font'))
        axis['name_font'] = self._convert_font_args(options.get('name_font'))
        axis['name_layout'] = self._get_layout_properties(options.get('name_layout'), True)
        axis['line'] = Shape._get_line_properties(options.get('line'))
        axis['fill'] = Shape._get_fill_properties(options.get('fill'))
        axis['pattern'] = Shape._get_pattern_properties(options.get('pattern'))
        axis['gradient'] = Shape._get_gradient_properties(options.get('gradient'))
        if axis.get('pattern'):
            axis['fill'] = None
        if axis.get('gradient'):
            axis['pattern'] = None
            axis['fill'] = None
        axis['minor_tick_mark'] = self._get_tick_type(options.get('minor_tick_mark'))
        axis['major_tick_mark'] = self._get_tick_type(options.get('major_tick_mark'))
        return axis

    def _convert_font_args(self, options):
        if False:
            return 10
        if not options:
            return
        font = {'name': options.get('name'), 'color': options.get('color'), 'size': options.get('size'), 'bold': options.get('bold'), 'italic': options.get('italic'), 'underline': options.get('underline'), 'pitch_family': options.get('pitch_family'), 'charset': options.get('charset'), 'baseline': options.get('baseline', 0), 'rotation': options.get('rotation')}
        if font['size']:
            font['size'] = int(font['size'] * 100)
        if font['rotation']:
            font['rotation'] = 60000 * int(font['rotation'])
        return font

    def _list_to_formula(self, data):
        if False:
            while True:
                i = 10
        if not isinstance(data, list):
            if data and ' ' in data and ("'" not in data) and self.warn_sheetname:
                warn("Sheetname in '%s' contains spaces but isn't quoted. This may cause errors in Excel." % data)
            return data
        formula = xl_range_formula(*data)
        return formula

    def _process_names(self, name, name_formula):
        if False:
            for i in range(10):
                print('nop')
        if name is not None:
            if isinstance(name, list):
                cell = xl_rowcol_to_cell(name[1], name[2], True, True)
                name_formula = quote_sheetname(name[0]) + '!' + cell
                name = ''
            elif re.match('^=?[^!]+!\\$?[A-Z]+\\$?\\d+', name):
                name_formula = name
                name = ''
        return (name, name_formula)

    def _get_data_type(self, data):
        if False:
            while True:
                i = 10
        if data is None or len(data) == 0:
            return 'none'
        if isinstance(data[0], list):
            return 'multi_str'
        for token in data:
            if token is None:
                continue
            if isinstance(token, str) and re.search('[_ ]', token):
                return 'str'
            try:
                float(token)
            except ValueError:
                return 'str'
        return 'num'

    def _get_data_id(self, formula, data):
        if False:
            while True:
                i = 10
        if not formula:
            return
        if formula.startswith('='):
            formula = formula.lstrip('=')
        if formula not in self.formula_ids:
            formula_id = len(self.formula_data)
            self.formula_data.append(data)
            self.formula_ids[formula] = formula_id
        else:
            formula_id = self.formula_ids[formula]
            if self.formula_data[formula_id] is None:
                self.formula_data[formula_id] = data
        return formula_id

    def _get_marker_properties(self, marker):
        if False:
            return 10
        if not marker:
            return
        marker = copy.deepcopy(marker)
        types = {'automatic': 'automatic', 'none': 'none', 'square': 'square', 'diamond': 'diamond', 'triangle': 'triangle', 'x': 'x', 'star': 'star', 'dot': 'dot', 'short_dash': 'dot', 'dash': 'dash', 'long_dash': 'dash', 'circle': 'circle', 'plus': 'plus', 'picture': 'picture'}
        marker_type = marker.get('type')
        if marker_type is not None:
            if marker_type in types:
                marker['type'] = types[marker_type]
            else:
                warn("Unknown marker type '%s" % marker_type)
                return
        line = Shape._get_line_properties(marker.get('line'))
        if 'border' in marker:
            line = Shape._get_line_properties(marker['border'])
        fill = Shape._get_fill_properties(marker.get('fill'))
        pattern = Shape._get_pattern_properties(marker.get('pattern'))
        gradient = Shape._get_gradient_properties(marker.get('gradient'))
        if pattern:
            self.fill = None
        if gradient:
            pattern = None
            fill = None
        marker['line'] = line
        marker['fill'] = fill
        marker['pattern'] = pattern
        marker['gradient'] = gradient
        return marker

    def _get_trendline_properties(self, trendline):
        if False:
            i = 10
            return i + 15
        if not trendline:
            return
        trendline = copy.deepcopy(trendline)
        types = {'exponential': 'exp', 'linear': 'linear', 'log': 'log', 'moving_average': 'movingAvg', 'polynomial': 'poly', 'power': 'power'}
        trend_type = trendline.get('type')
        if trend_type in types:
            trendline['type'] = types[trend_type]
        else:
            warn("Unknown trendline type '%s'" % trend_type)
            return
        line = Shape._get_line_properties(trendline.get('line'))
        if 'border' in trendline:
            line = Shape._get_line_properties(trendline['border'])
        fill = Shape._get_fill_properties(trendline.get('fill'))
        pattern = Shape._get_pattern_properties(trendline.get('pattern'))
        gradient = Shape._get_gradient_properties(trendline.get('gradient'))
        label = self._get_trendline_label_properties(trendline.get('label'))
        if pattern:
            self.fill = None
        if gradient:
            pattern = None
            fill = None
        trendline['line'] = line
        trendline['fill'] = fill
        trendline['pattern'] = pattern
        trendline['gradient'] = gradient
        trendline['label'] = label
        return trendline

    def _get_trendline_label_properties(self, label):
        if False:
            while True:
                i = 10
        if not label:
            return {}
        label = copy.deepcopy(label)
        font = self._convert_font_args(label.get('font'))
        line = Shape._get_line_properties(label.get('line'))
        if 'border' in label:
            line = Shape._get_line_properties(label['border'])
        fill = Shape._get_fill_properties(label.get('fill'))
        pattern = Shape._get_pattern_properties(label.get('pattern'))
        gradient = Shape._get_gradient_properties(label.get('gradient'))
        if pattern:
            self.fill = None
        if gradient:
            pattern = None
            fill = None
        label['font'] = font
        label['line'] = line
        label['fill'] = fill
        label['pattern'] = pattern
        label['gradient'] = gradient
        return label

    def _get_error_bars_props(self, options):
        if False:
            i = 10
            return i + 15
        if not options:
            return
        error_bars = {'type': 'fixedVal', 'value': 1, 'endcap': 1, 'direction': 'both'}
        types = {'fixed': 'fixedVal', 'percentage': 'percentage', 'standard_deviation': 'stdDev', 'standard_error': 'stdErr', 'custom': 'cust'}
        error_type = options['type']
        if error_type in types:
            error_bars['type'] = types[error_type]
        else:
            warn("Unknown error bars type '%s" % error_type)
            return
        if 'value' in options:
            error_bars['value'] = options['value']
        if 'end_style' in options:
            error_bars['endcap'] = options['end_style']
        if 'direction' in options:
            if options['direction'] == 'minus':
                error_bars['direction'] = 'minus'
            elif options['direction'] == 'plus':
                error_bars['direction'] = 'plus'
            else:
                pass
        error_bars['plus_values'] = options.get('plus_values')
        error_bars['minus_values'] = options.get('minus_values')
        error_bars['plus_data'] = options.get('plus_data')
        error_bars['minus_data'] = options.get('minus_data')
        error_bars['line'] = Shape._get_line_properties(options.get('line'))
        return error_bars

    def _get_gridline_properties(self, options):
        if False:
            return 10
        gridline = {'visible': options.get('visible')}
        gridline['line'] = Shape._get_line_properties(options.get('line'))
        return gridline

    def _get_labels_properties(self, labels):
        if False:
            i = 10
            return i + 15
        if not labels:
            return None
        labels = copy.deepcopy(labels)
        position = labels.get('position')
        if position:
            if position in self.label_positions:
                if position == self.label_position_default:
                    labels['position'] = None
                else:
                    labels['position'] = self.label_positions[position]
            else:
                warn("Unsupported label position '%s' for this chart type" % position)
                return
        separator = labels.get('separator')
        separators = {',': ', ', ';': '; ', '.': '. ', '\n': '\n', ' ': ' '}
        if separator:
            if separator in separators:
                labels['separator'] = separators[separator]
            else:
                warn('Unsupported label separator')
                return
        labels['font'] = self._convert_font_args(labels.get('font'))
        line = Shape._get_line_properties(labels.get('line'))
        if 'border' in labels:
            line = Shape._get_line_properties(labels['border'])
        fill = Shape._get_fill_properties(labels.get('fill'))
        pattern = Shape._get_pattern_properties(labels.get('pattern'))
        gradient = Shape._get_gradient_properties(labels.get('gradient'))
        if pattern:
            self.fill = None
        if gradient:
            pattern = None
            fill = None
        labels['line'] = line
        labels['fill'] = fill
        labels['pattern'] = pattern
        labels['gradient'] = gradient
        if labels.get('custom'):
            for label in labels['custom']:
                if label is None:
                    continue
                value = label.get('value')
                if value and re.match('^=?[^!]+!\\$?[A-Z]+\\$?\\d+', str(value)):
                    label['formula'] = value
                formula = label.get('formula')
                if formula and formula.startswith('='):
                    label['formula'] = formula.lstrip('=')
                data_id = self._get_data_id(formula, label.get('data'))
                label['data_id'] = data_id
                label['font'] = self._convert_font_args(label.get('font'))
                line = Shape._get_line_properties(label.get('line'))
                if 'border' in label:
                    line = Shape._get_line_properties(label['border'])
                fill = Shape._get_fill_properties(label.get('fill'))
                pattern = Shape._get_pattern_properties(label.get('pattern'))
                gradient = Shape._get_gradient_properties(label.get('gradient'))
                if pattern:
                    self.fill = None
                if gradient:
                    pattern = None
                    fill = None
                label['line'] = line
                label['fill'] = fill
                label['pattern'] = pattern
                label['gradient'] = gradient
        return labels

    def _get_area_properties(self, options):
        if False:
            for i in range(10):
                print('nop')
        area = {}
        line = Shape._get_line_properties(options.get('line'))
        if options.get('border'):
            line = Shape._get_line_properties(options['border'])
        fill = Shape._get_fill_properties(options.get('fill'))
        pattern = Shape._get_pattern_properties(options.get('pattern'))
        gradient = Shape._get_gradient_properties(options.get('gradient'))
        if pattern:
            self.fill = None
        if gradient:
            pattern = None
            fill = None
        layout = self._get_layout_properties(options.get('layout'), False)
        area['line'] = line
        area['fill'] = fill
        area['pattern'] = pattern
        area['layout'] = layout
        area['gradient'] = gradient
        return area

    def _get_legend_properties(self, options=None):
        if False:
            i = 10
            return i + 15
        legend = {}
        if options is None:
            options = {}
        legend['position'] = options.get('position', 'right')
        legend['delete_series'] = options.get('delete_series')
        legend['font'] = self._convert_font_args(options.get('font'))
        legend['layout'] = self._get_layout_properties(options.get('layout'), False)
        if options.get('none'):
            legend['position'] = 'none'
        line = Shape._get_line_properties(options.get('line'))
        if options.get('border'):
            line = Shape._get_line_properties(options['border'])
        fill = Shape._get_fill_properties(options.get('fill'))
        pattern = Shape._get_pattern_properties(options.get('pattern'))
        gradient = Shape._get_gradient_properties(options.get('gradient'))
        if pattern:
            self.fill = None
        if gradient:
            pattern = None
            fill = None
        layout = self._get_layout_properties(options.get('layout'), False)
        legend['line'] = line
        legend['fill'] = fill
        legend['pattern'] = pattern
        legend['layout'] = layout
        legend['gradient'] = gradient
        return legend

    def _get_layout_properties(self, args, is_text):
        if False:
            i = 10
            return i + 15
        layout = {}
        if not args:
            return
        if is_text:
            properties = ('x', 'y')
        else:
            properties = ('x', 'y', 'width', 'height')
        for key in args.keys():
            if key not in properties:
                warn("Property '%s' allowed not in layout options" % key)
                return
        for prop in properties:
            if prop not in args.keys():
                warn("Property '%s' must be specified in layout options" % prop)
                return
            value = args[prop]
            try:
                float(value)
            except ValueError:
                warn("Property '%s' value '%s' must be numeric in layout" % (prop, value))
                return
            if value < 0 or value > 1:
                warn("Property '%s' value '%s' must be in range 0 < x <= 1 in layout options" % (prop, value))
                return
            layout[prop] = '%.17g' % value
        return layout

    def _get_points_properties(self, user_points):
        if False:
            print('Hello World!')
        points = []
        if not user_points:
            return
        for user_point in user_points:
            point = {}
            if user_point is not None:
                line = Shape._get_line_properties(user_point.get('line'))
                if 'border' in user_point:
                    line = Shape._get_line_properties(user_point['border'])
                fill = Shape._get_fill_properties(user_point.get('fill'))
                pattern = Shape._get_pattern_properties(user_point.get('pattern'))
                gradient = Shape._get_gradient_properties(user_point.get('gradient'))
                if pattern:
                    self.fill = None
                if gradient:
                    pattern = None
                    fill = None
                point['line'] = line
                point['fill'] = fill
                point['pattern'] = pattern
                point['gradient'] = gradient
            points.append(point)
        return points

    def _has_fill_formatting(self, element):
        if False:
            i = 10
            return i + 15
        has_fill = False
        has_line = False
        has_pattern = element.get('pattern')
        has_gradient = element.get('gradient')
        if element.get('fill') and element['fill']['defined']:
            has_fill = True
        if element.get('line') and element['line']['defined']:
            has_line = True
        if not has_fill and (not has_line) and (not has_pattern) and (not has_gradient):
            return False
        else:
            return True

    def _get_display_units(self, display_units):
        if False:
            return 10
        if not display_units:
            return
        types = {'hundreds': 'hundreds', 'thousands': 'thousands', 'ten_thousands': 'tenThousands', 'hundred_thousands': 'hundredThousands', 'millions': 'millions', 'ten_millions': 'tenMillions', 'hundred_millions': 'hundredMillions', 'billions': 'billions', 'trillions': 'trillions'}
        if display_units in types:
            display_units = types[display_units]
        else:
            warn("Unknown display_units type '%s'" % display_units)
            return
        return display_units

    def _get_tick_type(self, tick_type):
        if False:
            i = 10
            return i + 15
        if not tick_type:
            return
        types = {'outside': 'out', 'inside': 'in', 'none': 'none', 'cross': 'cross'}
        if tick_type in types:
            tick_type = types[tick_type]
        else:
            warn("Unknown tick_type  '%s'" % tick_type)
            return
        return tick_type

    def _get_primary_axes_series(self):
        if False:
            return 10
        primary_axes_series = []
        for series in self.series:
            if not series['y2_axis']:
                primary_axes_series.append(series)
        return primary_axes_series

    def _get_secondary_axes_series(self):
        if False:
            while True:
                i = 10
        secondary_axes_series = []
        for series in self.series:
            if series['y2_axis']:
                secondary_axes_series.append(series)
        return secondary_axes_series

    def _add_axis_ids(self, args):
        if False:
            print('Hello World!')
        chart_id = 5001 + int(self.id)
        axis_count = 1 + len(self.axis2_ids) + len(self.axis_ids)
        id1 = '%04d%04d' % (chart_id, axis_count)
        id2 = '%04d%04d' % (chart_id, axis_count + 1)
        if args['primary_axes']:
            self.axis_ids.append(id1)
            self.axis_ids.append(id2)
        if not args['primary_axes']:
            self.axis2_ids.append(id1)
            self.axis2_ids.append(id2)

    def _set_default_properties(self):
        if False:
            i = 10
            return i + 15
        self.x_axis['defaults'] = {'num_format': 'General', 'major_gridlines': {'visible': 0}}
        self.y_axis['defaults'] = {'num_format': 'General', 'major_gridlines': {'visible': 1}}
        self.x2_axis['defaults'] = {'num_format': 'General', 'label_position': 'none', 'crossing': 'max', 'visible': 0}
        self.y2_axis['defaults'] = {'num_format': 'General', 'major_gridlines': {'visible': 0}, 'position': 'right', 'visible': 1}
        self.set_x_axis({})
        self.set_y_axis({})
        self.set_x2_axis({})
        self.set_y2_axis({})

    def _write_chart_space(self):
        if False:
            while True:
                i = 10
        schema = 'http://schemas.openxmlformats.org/'
        xmlns_c = schema + 'drawingml/2006/chart'
        xmlns_a = schema + 'drawingml/2006/main'
        xmlns_r = schema + 'officeDocument/2006/relationships'
        attributes = [('xmlns:c', xmlns_c), ('xmlns:a', xmlns_a), ('xmlns:r', xmlns_r)]
        self._xml_start_tag('c:chartSpace', attributes)

    def _write_lang(self):
        if False:
            while True:
                i = 10
        val = 'en-US'
        attributes = [('val', val)]
        self._xml_empty_tag('c:lang', attributes)

    def _write_style(self):
        if False:
            print('Hello World!')
        style_id = self.style_id
        if style_id == 2:
            return
        attributes = [('val', style_id)]
        self._xml_empty_tag('c:style', attributes)

    def _write_chart(self):
        if False:
            return 10
        self._xml_start_tag('c:chart')
        if self.title_none:
            self._write_c_auto_title_deleted()
        elif self.title_formula is not None:
            self._write_title_formula(self.title_formula, self.title_data_id, None, self.title_font, self.title_layout, self.title_overlay)
        elif self.title_name is not None:
            self._write_title_rich(self.title_name, None, self.title_font, self.title_layout, self.title_overlay)
        self._write_plot_area()
        self._write_legend()
        self._write_plot_vis_only()
        self._write_disp_blanks_as()
        if self.show_na_as_empty:
            self._write_c_ext_lst_display_na()
        self._xml_end_tag('c:chart')

    def _write_disp_blanks_as(self):
        if False:
            i = 10
            return i + 15
        val = self.show_blanks
        if val == 'gap':
            return
        attributes = [('val', val)]
        self._xml_empty_tag('c:dispBlanksAs', attributes)

    def _write_plot_area(self):
        if False:
            while True:
                i = 10
        self._xml_start_tag('c:plotArea')
        self._write_layout(self.plotarea.get('layout'), 'plot')
        self._write_chart_type({'primary_axes': True})
        self._write_chart_type({'primary_axes': False})
        second_chart = self.combined
        if second_chart:
            if second_chart.is_secondary:
                second_chart.id = 1000 + self.id
            else:
                second_chart.id = self.id
            second_chart.fh = self.fh
            second_chart.series_index = self.series_index
            second_chart._write_chart_type({'primary_axes': True})
            second_chart._write_chart_type({'primary_axes': False})
        args = {'x_axis': self.x_axis, 'y_axis': self.y_axis, 'axis_ids': self.axis_ids}
        if self.date_category:
            self._write_date_axis(args)
        else:
            self._write_cat_axis(args)
        self._write_val_axis(args)
        args = {'x_axis': self.x2_axis, 'y_axis': self.y2_axis, 'axis_ids': self.axis2_ids}
        self._write_val_axis(args)
        if second_chart and second_chart.is_secondary:
            args = {'x_axis': second_chart.x2_axis, 'y_axis': second_chart.y2_axis, 'axis_ids': second_chart.axis2_ids}
            second_chart._write_val_axis(args)
        if self.date_category:
            self._write_date_axis(args)
        else:
            self._write_cat_axis(args)
        self._write_d_table()
        self._write_sp_pr(self.plotarea)
        self._xml_end_tag('c:plotArea')

    def _write_layout(self, layout, layout_type):
        if False:
            while True:
                i = 10
        if not layout:
            self._xml_empty_tag('c:layout')
        else:
            self._xml_start_tag('c:layout')
            self._write_manual_layout(layout, layout_type)
            self._xml_end_tag('c:layout')

    def _write_manual_layout(self, layout, layout_type):
        if False:
            print('Hello World!')
        self._xml_start_tag('c:manualLayout')
        if layout_type == 'plot':
            self._xml_empty_tag('c:layoutTarget', [('val', 'inner')])
        self._xml_empty_tag('c:xMode', [('val', 'edge')])
        self._xml_empty_tag('c:yMode', [('val', 'edge')])
        self._xml_empty_tag('c:x', [('val', layout['x'])])
        self._xml_empty_tag('c:y', [('val', layout['y'])])
        if layout_type != 'text':
            self._xml_empty_tag('c:w', [('val', layout['width'])])
            self._xml_empty_tag('c:h', [('val', layout['height'])])
        self._xml_end_tag('c:manualLayout')

    def _write_chart_type(self, options):
        if False:
            i = 10
            return i + 15
        return

    def _write_grouping(self, val):
        if False:
            for i in range(10):
                print('nop')
        attributes = [('val', val)]
        self._xml_empty_tag('c:grouping', attributes)

    def _write_series(self, series):
        if False:
            return 10
        self._write_ser(series)

    def _write_ser(self, series):
        if False:
            return 10
        index = self.series_index
        self.series_index += 1
        self._xml_start_tag('c:ser')
        self._write_idx(index)
        self._write_order(index)
        self._write_series_name(series)
        self._write_sp_pr(series)
        self._write_marker(series['marker'])
        self._write_c_invert_if_negative(series['invert_if_neg'])
        self._write_d_pt(series['points'])
        self._write_d_lbls(series['labels'])
        self._write_trendline(series['trendline'])
        self._write_error_bars(series['error_bars'])
        self._write_cat(series)
        self._write_val(series)
        if self.smooth_allowed:
            self._write_c_smooth(series['smooth'])
        if series.get('inverted_color'):
            self._write_c_ext_lst_inverted_color(series['inverted_color'])
        self._xml_end_tag('c:ser')

    def _write_c_ext_lst_inverted_color(self, color):
        if False:
            print('Hello World!')
        uri = '{6F2FDCE9-48DA-4B69-8628-5D25D57E5C99}'
        xmlns_c_14 = 'http://schemas.microsoft.com/office/drawing/2007/8/2/chart'
        attributes1 = [('uri', uri), ('xmlns:c14', xmlns_c_14)]
        attributes2 = [('xmlns:c14', xmlns_c_14)]
        self._xml_start_tag('c:extLst')
        self._xml_start_tag('c:ext', attributes1)
        self._xml_start_tag('c14:invertSolidFillFmt')
        self._xml_start_tag('c14:spPr', attributes2)
        self._write_a_solid_fill({'color': color})
        self._xml_end_tag('c14:spPr')
        self._xml_end_tag('c14:invertSolidFillFmt')
        self._xml_end_tag('c:ext')
        self._xml_end_tag('c:extLst')

    def _write_c_ext_lst_display_na(self):
        if False:
            while True:
                i = 10
        uri = '{56B9EC1D-385E-4148-901F-78D8002777C0}'
        xmlns_c_16 = 'http://schemas.microsoft.com/office/drawing/2017/03/chart'
        attributes1 = [('uri', uri), ('xmlns:c16r3', xmlns_c_16)]
        attributes2 = [('val', 1)]
        self._xml_start_tag('c:extLst')
        self._xml_start_tag('c:ext', attributes1)
        self._xml_start_tag('c16r3:dataDisplayOptions16')
        self._xml_empty_tag('c16r3:dispNaAsBlank', attributes2)
        self._xml_end_tag('c16r3:dataDisplayOptions16')
        self._xml_end_tag('c:ext')
        self._xml_end_tag('c:extLst')

    def _write_idx(self, val):
        if False:
            for i in range(10):
                print('nop')
        attributes = [('val', val)]
        self._xml_empty_tag('c:idx', attributes)

    def _write_order(self, val):
        if False:
            i = 10
            return i + 15
        attributes = [('val', val)]
        self._xml_empty_tag('c:order', attributes)

    def _write_series_name(self, series):
        if False:
            for i in range(10):
                print('nop')
        if series['name_formula'] is not None:
            self._write_tx_formula(series['name_formula'], series['name_id'])
        elif series['name'] is not None:
            self._write_tx_value(series['name'])

    def _write_c_smooth(self, smooth):
        if False:
            i = 10
            return i + 15
        if smooth:
            self._xml_empty_tag('c:smooth', [('val', '1')])

    def _write_cat(self, series):
        if False:
            print('Hello World!')
        formula = series['categories']
        data_id = series['cat_data_id']
        data = None
        if data_id is not None:
            data = self.formula_data[data_id]
        if not formula:
            return
        self._xml_start_tag('c:cat')
        cat_type = self._get_data_type(data)
        if cat_type == 'str':
            self.cat_has_num_fmt = 0
            self._write_str_ref(formula, data, cat_type)
        elif cat_type == 'multi_str':
            self.cat_has_num_fmt = 0
            self._write_multi_lvl_str_ref(formula, data)
        else:
            self.cat_has_num_fmt = 1
            self._write_num_ref(formula, data, cat_type)
        self._xml_end_tag('c:cat')

    def _write_val(self, series):
        if False:
            while True:
                i = 10
        formula = series['values']
        data_id = series['val_data_id']
        data = self.formula_data[data_id]
        self._xml_start_tag('c:val')
        self._write_num_ref(formula, data, 'num')
        self._xml_end_tag('c:val')

    def _write_num_ref(self, formula, data, ref_type):
        if False:
            return 10
        self._xml_start_tag('c:numRef')
        self._write_series_formula(formula)
        if ref_type == 'num':
            self._write_num_cache(data)
        elif ref_type == 'str':
            self._write_str_cache(data)
        self._xml_end_tag('c:numRef')

    def _write_str_ref(self, formula, data, ref_type):
        if False:
            print('Hello World!')
        self._xml_start_tag('c:strRef')
        self._write_series_formula(formula)
        if ref_type == 'num':
            self._write_num_cache(data)
        elif ref_type == 'str':
            self._write_str_cache(data)
        self._xml_end_tag('c:strRef')

    def _write_multi_lvl_str_ref(self, formula, data):
        if False:
            return 10
        if not data:
            return
        self._xml_start_tag('c:multiLvlStrRef')
        self._write_series_formula(formula)
        self._xml_start_tag('c:multiLvlStrCache')
        count = len(data[-1])
        self._write_pt_count(count)
        for cat_data in reversed(data):
            self._xml_start_tag('c:lvl')
            for (i, point) in enumerate(cat_data):
                self._write_pt(i, cat_data[i])
            self._xml_end_tag('c:lvl')
        self._xml_end_tag('c:multiLvlStrCache')
        self._xml_end_tag('c:multiLvlStrRef')

    def _write_series_formula(self, formula):
        if False:
            i = 10
            return i + 15
        if formula.startswith('='):
            formula = formula.lstrip('=')
        self._xml_data_element('c:f', formula)

    def _write_axis_ids(self, args):
        if False:
            return 10
        self._add_axis_ids(args)
        if args['primary_axes']:
            self._write_axis_id(self.axis_ids[0])
            self._write_axis_id(self.axis_ids[1])
        else:
            self._write_axis_id(self.axis2_ids[0])
            self._write_axis_id(self.axis2_ids[1])

    def _write_axis_id(self, val):
        if False:
            return 10
        attributes = [('val', val)]
        self._xml_empty_tag('c:axId', attributes)

    def _write_cat_axis(self, args):
        if False:
            while True:
                i = 10
        x_axis = args['x_axis']
        y_axis = args['y_axis']
        axis_ids = args['axis_ids']
        if axis_ids is None or not len(axis_ids):
            return
        position = self.cat_axis_position
        is_y_axis = self.horiz_cat_axis
        if x_axis.get('position'):
            position = x_axis['position']
        self._xml_start_tag('c:catAx')
        self._write_axis_id(axis_ids[0])
        self._write_scaling(x_axis.get('reverse'), None, None, None)
        if not x_axis.get('visible'):
            self._write_delete(1)
        self._write_axis_pos(position, y_axis.get('reverse'))
        self._write_major_gridlines(x_axis.get('major_gridlines'))
        self._write_minor_gridlines(x_axis.get('minor_gridlines'))
        if x_axis['formula'] is not None:
            self._write_title_formula(x_axis['formula'], x_axis['data_id'], is_y_axis, x_axis['name_font'], x_axis['name_layout'])
        elif x_axis['name'] is not None:
            self._write_title_rich(x_axis['name'], is_y_axis, x_axis['name_font'], x_axis['name_layout'])
        self._write_cat_number_format(x_axis)
        self._write_major_tick_mark(x_axis.get('major_tick_mark'))
        self._write_minor_tick_mark(x_axis.get('minor_tick_mark'))
        self._write_tick_label_pos(x_axis.get('label_position'))
        self._write_sp_pr(x_axis)
        self._write_axis_font(x_axis.get('num_font'))
        self._write_cross_axis(axis_ids[1])
        if self.show_crosses or x_axis.get('visible'):
            if y_axis.get('crossing') is None or y_axis.get('crossing') == 'max' or y_axis['crossing'] == 'min':
                self._write_crosses(y_axis.get('crossing'))
            else:
                self._write_c_crosses_at(y_axis.get('crossing'))
        if not x_axis.get('text_axis'):
            self._write_auto(1)
        self._write_label_align(x_axis.get('label_align'))
        self._write_label_offset(100)
        self._write_c_tick_lbl_skip(x_axis.get('interval_unit'))
        self._write_c_tick_mark_skip(x_axis.get('interval_tick'))
        self._xml_end_tag('c:catAx')

    def _write_val_axis(self, args):
        if False:
            for i in range(10):
                print('nop')
        x_axis = args['x_axis']
        y_axis = args['y_axis']
        axis_ids = args['axis_ids']
        position = args.get('position', self.val_axis_position)
        is_y_axis = self.horiz_val_axis
        if axis_ids is None or not len(axis_ids):
            return
        position = y_axis.get('position') or position
        self._xml_start_tag('c:valAx')
        self._write_axis_id(axis_ids[1])
        self._write_scaling(y_axis.get('reverse'), y_axis.get('min'), y_axis.get('max'), y_axis.get('log_base'))
        if not y_axis.get('visible'):
            self._write_delete(1)
        self._write_axis_pos(position, x_axis.get('reverse'))
        self._write_major_gridlines(y_axis.get('major_gridlines'))
        self._write_minor_gridlines(y_axis.get('minor_gridlines'))
        if y_axis['formula'] is not None:
            self._write_title_formula(y_axis['formula'], y_axis['data_id'], is_y_axis, y_axis['name_font'], y_axis['name_layout'])
        elif y_axis['name'] is not None:
            self._write_title_rich(y_axis['name'], is_y_axis, y_axis.get('name_font'), y_axis.get('name_layout'))
        self._write_number_format(y_axis)
        self._write_major_tick_mark(y_axis.get('major_tick_mark'))
        self._write_minor_tick_mark(y_axis.get('minor_tick_mark'))
        self._write_tick_label_pos(y_axis.get('label_position'))
        self._write_sp_pr(y_axis)
        self._write_axis_font(y_axis.get('num_font'))
        self._write_cross_axis(axis_ids[0])
        if x_axis.get('crossing') is None or x_axis['crossing'] == 'max' or x_axis['crossing'] == 'min':
            self._write_crosses(x_axis.get('crossing'))
        else:
            self._write_c_crosses_at(x_axis.get('crossing'))
        self._write_cross_between(x_axis.get('position_axis'))
        self._write_c_major_unit(y_axis.get('major_unit'))
        self._write_c_minor_unit(y_axis.get('minor_unit'))
        self._write_disp_units(y_axis.get('display_units'), y_axis.get('display_units_visible'))
        self._xml_end_tag('c:valAx')

    def _write_cat_val_axis(self, args):
        if False:
            return 10
        x_axis = args['x_axis']
        y_axis = args['y_axis']
        axis_ids = args['axis_ids']
        position = args['position'] or self.val_axis_position
        is_y_axis = self.horiz_val_axis
        if axis_ids is None or not len(axis_ids):
            return
        position = x_axis.get('position') or position
        self._xml_start_tag('c:valAx')
        self._write_axis_id(axis_ids[0])
        self._write_scaling(x_axis.get('reverse'), x_axis.get('min'), x_axis.get('max'), x_axis.get('log_base'))
        if not x_axis.get('visible'):
            self._write_delete(1)
        self._write_axis_pos(position, y_axis.get('reverse'))
        self._write_major_gridlines(x_axis.get('major_gridlines'))
        self._write_minor_gridlines(x_axis.get('minor_gridlines'))
        if x_axis['formula'] is not None:
            self._write_title_formula(x_axis['formula'], x_axis['data_id'], is_y_axis, x_axis['name_font'], x_axis['name_layout'])
        elif x_axis['name'] is not None:
            self._write_title_rich(x_axis['name'], is_y_axis, x_axis['name_font'], x_axis['name_layout'])
        self._write_number_format(x_axis)
        self._write_major_tick_mark(x_axis.get('major_tick_mark'))
        self._write_minor_tick_mark(x_axis.get('minor_tick_mark'))
        self._write_tick_label_pos(x_axis.get('label_position'))
        self._write_sp_pr(x_axis)
        self._write_axis_font(x_axis.get('num_font'))
        self._write_cross_axis(axis_ids[1])
        if y_axis.get('crossing') is None or y_axis['crossing'] == 'max' or y_axis['crossing'] == 'min':
            self._write_crosses(y_axis.get('crossing'))
        else:
            self._write_c_crosses_at(y_axis.get('crossing'))
        self._write_cross_between(y_axis.get('position_axis'))
        self._write_c_major_unit(x_axis.get('major_unit'))
        self._write_c_minor_unit(x_axis.get('minor_unit'))
        self._write_disp_units(x_axis.get('display_units'), x_axis.get('display_units_visible'))
        self._xml_end_tag('c:valAx')

    def _write_date_axis(self, args):
        if False:
            print('Hello World!')
        x_axis = args['x_axis']
        y_axis = args['y_axis']
        axis_ids = args['axis_ids']
        if axis_ids is None or not len(axis_ids):
            return
        position = self.cat_axis_position
        position = x_axis.get('position') or position
        self._xml_start_tag('c:dateAx')
        self._write_axis_id(axis_ids[0])
        self._write_scaling(x_axis.get('reverse'), x_axis.get('min'), x_axis.get('max'), x_axis.get('log_base'))
        if not x_axis.get('visible'):
            self._write_delete(1)
        self._write_axis_pos(position, y_axis.get('reverse'))
        self._write_major_gridlines(x_axis.get('major_gridlines'))
        self._write_minor_gridlines(x_axis.get('minor_gridlines'))
        if x_axis['formula'] is not None:
            self._write_title_formula(x_axis['formula'], x_axis['data_id'], None, x_axis['name_font'], x_axis['name_layout'])
        elif x_axis['name'] is not None:
            self._write_title_rich(x_axis['name'], None, x_axis['name_font'], x_axis['name_layout'])
        self._write_number_format(x_axis)
        self._write_major_tick_mark(x_axis.get('major_tick_mark'))
        self._write_minor_tick_mark(x_axis.get('minor_tick_mark'))
        self._write_tick_label_pos(x_axis.get('label_position'))
        self._write_sp_pr(x_axis)
        self._write_axis_font(x_axis.get('num_font'))
        self._write_cross_axis(axis_ids[1])
        if self.show_crosses or x_axis.get('visible'):
            if y_axis.get('crossing') is None or y_axis.get('crossing') == 'max' or y_axis['crossing'] == 'min':
                self._write_crosses(y_axis.get('crossing'))
            else:
                self._write_c_crosses_at(y_axis.get('crossing'))
        self._write_auto(1)
        self._write_label_offset(100)
        self._write_c_tick_lbl_skip(x_axis.get('interval_unit'))
        self._write_c_tick_mark_skip(x_axis.get('interval_tick'))
        self._write_c_major_unit(x_axis.get('major_unit'))
        if x_axis.get('major_unit'):
            self._write_c_major_time_unit(x_axis['major_unit_type'])
        self._write_c_minor_unit(x_axis.get('minor_unit'))
        if x_axis.get('minor_unit'):
            self._write_c_minor_time_unit(x_axis['minor_unit_type'])
        self._xml_end_tag('c:dateAx')

    def _write_scaling(self, reverse, min_val, max_val, log_base):
        if False:
            for i in range(10):
                print('nop')
        self._xml_start_tag('c:scaling')
        self._write_c_log_base(log_base)
        self._write_orientation(reverse)
        self._write_c_max(max_val)
        self._write_c_min(min_val)
        self._xml_end_tag('c:scaling')

    def _write_c_log_base(self, val):
        if False:
            print('Hello World!')
        if not val:
            return
        attributes = [('val', val)]
        self._xml_empty_tag('c:logBase', attributes)

    def _write_orientation(self, reverse):
        if False:
            return 10
        val = 'minMax'
        if reverse:
            val = 'maxMin'
        attributes = [('val', val)]
        self._xml_empty_tag('c:orientation', attributes)

    def _write_c_max(self, max_val):
        if False:
            for i in range(10):
                print('nop')
        if max_val is None:
            return
        attributes = [('val', max_val)]
        self._xml_empty_tag('c:max', attributes)

    def _write_c_min(self, min_val):
        if False:
            return 10
        if min_val is None:
            return
        attributes = [('val', min_val)]
        self._xml_empty_tag('c:min', attributes)

    def _write_axis_pos(self, val, reverse):
        if False:
            while True:
                i = 10
        if reverse:
            if val == 'l':
                val = 'r'
            if val == 'b':
                val = 't'
        attributes = [('val', val)]
        self._xml_empty_tag('c:axPos', attributes)

    def _write_number_format(self, axis):
        if False:
            return 10
        format_code = axis.get('num_format')
        source_linked = 1
        if format_code is not None and format_code != axis['defaults']['num_format']:
            source_linked = 0
        if axis.get('num_format_linked'):
            source_linked = 1
        attributes = [('formatCode', format_code), ('sourceLinked', source_linked)]
        self._xml_empty_tag('c:numFmt', attributes)

    def _write_cat_number_format(self, axis):
        if False:
            print('Hello World!')
        format_code = axis.get('num_format')
        source_linked = 1
        default_format = 1
        if format_code is not None and format_code != axis['defaults']['num_format']:
            source_linked = 0
            default_format = 0
        if axis.get('num_format_linked'):
            source_linked = 1
        if not self.cat_has_num_fmt and default_format:
            return
        attributes = [('formatCode', format_code), ('sourceLinked', source_linked)]
        self._xml_empty_tag('c:numFmt', attributes)

    def _write_data_label_number_format(self, format_code):
        if False:
            while True:
                i = 10
        source_linked = 0
        attributes = [('formatCode', format_code), ('sourceLinked', source_linked)]
        self._xml_empty_tag('c:numFmt', attributes)

    def _write_major_tick_mark(self, val):
        if False:
            i = 10
            return i + 15
        if not val:
            return
        attributes = [('val', val)]
        self._xml_empty_tag('c:majorTickMark', attributes)

    def _write_minor_tick_mark(self, val):
        if False:
            return 10
        if not val:
            return
        attributes = [('val', val)]
        self._xml_empty_tag('c:minorTickMark', attributes)

    def _write_tick_label_pos(self, val=None):
        if False:
            for i in range(10):
                print('nop')
        if val is None or val == 'next_to':
            val = 'nextTo'
        attributes = [('val', val)]
        self._xml_empty_tag('c:tickLblPos', attributes)

    def _write_cross_axis(self, val):
        if False:
            i = 10
            return i + 15
        attributes = [('val', val)]
        self._xml_empty_tag('c:crossAx', attributes)

    def _write_crosses(self, val=None):
        if False:
            for i in range(10):
                print('nop')
        if val is None:
            val = 'autoZero'
        attributes = [('val', val)]
        self._xml_empty_tag('c:crosses', attributes)

    def _write_c_crosses_at(self, val):
        if False:
            return 10
        attributes = [('val', val)]
        self._xml_empty_tag('c:crossesAt', attributes)

    def _write_auto(self, val):
        if False:
            for i in range(10):
                print('nop')
        attributes = [('val', val)]
        self._xml_empty_tag('c:auto', attributes)

    def _write_label_align(self, val=None):
        if False:
            while True:
                i = 10
        if val is None:
            val = 'ctr'
        if val == 'right':
            val = 'r'
        if val == 'left':
            val = 'l'
        attributes = [('val', val)]
        self._xml_empty_tag('c:lblAlgn', attributes)

    def _write_label_offset(self, val):
        if False:
            for i in range(10):
                print('nop')
        attributes = [('val', val)]
        self._xml_empty_tag('c:lblOffset', attributes)

    def _write_c_tick_lbl_skip(self, val):
        if False:
            return 10
        if val is None:
            return
        attributes = [('val', val)]
        self._xml_empty_tag('c:tickLblSkip', attributes)

    def _write_c_tick_mark_skip(self, val):
        if False:
            print('Hello World!')
        if val is None:
            return
        attributes = [('val', val)]
        self._xml_empty_tag('c:tickMarkSkip', attributes)

    def _write_major_gridlines(self, gridlines):
        if False:
            return 10
        if not gridlines:
            return
        if not gridlines['visible']:
            return
        if gridlines['line']['defined']:
            self._xml_start_tag('c:majorGridlines')
            self._write_sp_pr(gridlines)
            self._xml_end_tag('c:majorGridlines')
        else:
            self._xml_empty_tag('c:majorGridlines')

    def _write_minor_gridlines(self, gridlines):
        if False:
            print('Hello World!')
        if not gridlines:
            return
        if not gridlines['visible']:
            return
        if gridlines['line']['defined']:
            self._xml_start_tag('c:minorGridlines')
            self._write_sp_pr(gridlines)
            self._xml_end_tag('c:minorGridlines')
        else:
            self._xml_empty_tag('c:minorGridlines')

    def _write_cross_between(self, val):
        if False:
            return 10
        if val is None:
            val = self.cross_between
        attributes = [('val', val)]
        self._xml_empty_tag('c:crossBetween', attributes)

    def _write_c_major_unit(self, val):
        if False:
            i = 10
            return i + 15
        if not val:
            return
        attributes = [('val', val)]
        self._xml_empty_tag('c:majorUnit', attributes)

    def _write_c_minor_unit(self, val):
        if False:
            i = 10
            return i + 15
        if not val:
            return
        attributes = [('val', val)]
        self._xml_empty_tag('c:minorUnit', attributes)

    def _write_c_major_time_unit(self, val=None):
        if False:
            while True:
                i = 10
        if val is None:
            val = 'days'
        attributes = [('val', val)]
        self._xml_empty_tag('c:majorTimeUnit', attributes)

    def _write_c_minor_time_unit(self, val=None):
        if False:
            while True:
                i = 10
        if val is None:
            val = 'days'
        attributes = [('val', val)]
        self._xml_empty_tag('c:minorTimeUnit', attributes)

    def _write_legend(self):
        if False:
            return 10
        legend = self.legend
        position = legend.get('position', 'right')
        font = legend.get('font')
        delete_series = []
        overlay = 0
        if legend.get('delete_series') and isinstance(legend['delete_series'], list):
            delete_series = legend['delete_series']
        if position.startswith('overlay_'):
            position = position.replace('overlay_', '')
            overlay = 1
        allowed = {'right': 'r', 'left': 'l', 'top': 't', 'bottom': 'b', 'top_right': 'tr'}
        if position == 'none':
            return
        if position not in allowed:
            return
        position = allowed[position]
        self._xml_start_tag('c:legend')
        self._write_legend_pos(position)
        for index in delete_series:
            self._write_legend_entry(index)
        self._write_layout(legend.get('layout'), 'legend')
        if overlay:
            self._write_overlay()
        if font:
            self._write_tx_pr(font)
        self._write_sp_pr(legend)
        self._xml_end_tag('c:legend')

    def _write_legend_pos(self, val):
        if False:
            while True:
                i = 10
        attributes = [('val', val)]
        self._xml_empty_tag('c:legendPos', attributes)

    def _write_legend_entry(self, index):
        if False:
            while True:
                i = 10
        self._xml_start_tag('c:legendEntry')
        self._write_idx(index)
        self._write_delete(1)
        self._xml_end_tag('c:legendEntry')

    def _write_overlay(self):
        if False:
            print('Hello World!')
        val = 1
        attributes = [('val', val)]
        self._xml_empty_tag('c:overlay', attributes)

    def _write_plot_vis_only(self):
        if False:
            i = 10
            return i + 15
        val = 1
        if self.show_hidden:
            return
        attributes = [('val', val)]
        self._xml_empty_tag('c:plotVisOnly', attributes)

    def _write_print_settings(self):
        if False:
            i = 10
            return i + 15
        self._xml_start_tag('c:printSettings')
        self._write_header_footer()
        self._write_page_margins()
        self._write_page_setup()
        self._xml_end_tag('c:printSettings')

    def _write_header_footer(self):
        if False:
            i = 10
            return i + 15
        self._xml_empty_tag('c:headerFooter')

    def _write_page_margins(self):
        if False:
            i = 10
            return i + 15
        bottom = 0.75
        left = 0.7
        right = 0.7
        top = 0.75
        header = 0.3
        footer = 0.3
        attributes = [('b', bottom), ('l', left), ('r', right), ('t', top), ('header', header), ('footer', footer)]
        self._xml_empty_tag('c:pageMargins', attributes)

    def _write_page_setup(self):
        if False:
            return 10
        self._xml_empty_tag('c:pageSetup')

    def _write_c_auto_title_deleted(self):
        if False:
            print('Hello World!')
        self._xml_empty_tag('c:autoTitleDeleted', [('val', 1)])

    def _write_title_rich(self, title, is_y_axis, font, layout, overlay=False):
        if False:
            i = 10
            return i + 15
        self._xml_start_tag('c:title')
        self._write_tx_rich(title, is_y_axis, font)
        self._write_layout(layout, 'text')
        if overlay:
            self._write_overlay()
        self._xml_end_tag('c:title')

    def _write_title_formula(self, title, data_id, is_y_axis, font, layout, overlay=False):
        if False:
            print('Hello World!')
        self._xml_start_tag('c:title')
        self._write_tx_formula(title, data_id)
        self._write_layout(layout, 'text')
        if overlay:
            self._write_overlay()
        self._write_tx_pr(font, is_y_axis)
        self._xml_end_tag('c:title')

    def _write_tx_rich(self, title, is_y_axis, font):
        if False:
            return 10
        self._xml_start_tag('c:tx')
        self._write_rich(title, font, is_y_axis, ignore_rich_pr=False)
        self._xml_end_tag('c:tx')

    def _write_tx_value(self, title):
        if False:
            print('Hello World!')
        self._xml_start_tag('c:tx')
        self._write_v(title)
        self._xml_end_tag('c:tx')

    def _write_tx_formula(self, title, data_id):
        if False:
            while True:
                i = 10
        data = None
        if data_id is not None:
            data = self.formula_data[data_id]
        self._xml_start_tag('c:tx')
        self._write_str_ref(title, data, 'str')
        self._xml_end_tag('c:tx')

    def _write_rich(self, title, font, is_y_axis, ignore_rich_pr):
        if False:
            i = 10
            return i + 15
        if font and font.get('rotation') is not None:
            rotation = font['rotation']
        else:
            rotation = None
        self._xml_start_tag('c:rich')
        self._write_a_body_pr(rotation, is_y_axis)
        self._write_a_lst_style()
        self._write_a_p_rich(title, font, ignore_rich_pr)
        self._xml_end_tag('c:rich')

    def _write_a_body_pr(self, rotation, is_y_axis):
        if False:
            for i in range(10):
                print('nop')
        attributes = []
        if rotation is None and is_y_axis:
            rotation = -5400000
        if rotation is not None:
            if rotation == 16200000:
                attributes.append(('rot', 0))
                attributes.append(('vert', 'wordArtVert'))
            elif rotation == 16260000:
                attributes.append(('rot', 0))
                attributes.append(('vert', 'eaVert'))
            else:
                attributes.append(('rot', rotation))
                attributes.append(('vert', 'horz'))
        self._xml_empty_tag('a:bodyPr', attributes)

    def _write_a_lst_style(self):
        if False:
            return 10
        self._xml_empty_tag('a:lstStyle')

    def _write_a_p_rich(self, title, font, ignore_rich_pr):
        if False:
            return 10
        self._xml_start_tag('a:p')
        if not ignore_rich_pr:
            self._write_a_p_pr_rich(font)
        self._write_a_r(title, font)
        self._xml_end_tag('a:p')

    def _write_a_p_formula(self, font):
        if False:
            for i in range(10):
                print('nop')
        self._xml_start_tag('a:p')
        self._write_a_p_pr_rich(font)
        self._write_a_end_para_rpr()
        self._xml_end_tag('a:p')

    def _write_a_p_pr_rich(self, font):
        if False:
            return 10
        self._xml_start_tag('a:pPr')
        self._write_a_def_rpr(font)
        self._xml_end_tag('a:pPr')

    def _write_a_def_rpr(self, font):
        if False:
            for i in range(10):
                print('nop')
        has_color = 0
        style_attributes = Shape._get_font_style_attributes(font)
        latin_attributes = Shape._get_font_latin_attributes(font)
        if font and font.get('color') is not None:
            has_color = 1
        if latin_attributes or has_color:
            self._xml_start_tag('a:defRPr', style_attributes)
            if has_color:
                self._write_a_solid_fill({'color': font['color']})
            if latin_attributes:
                self._write_a_latin(latin_attributes)
            self._xml_end_tag('a:defRPr')
        else:
            self._xml_empty_tag('a:defRPr', style_attributes)

    def _write_a_end_para_rpr(self):
        if False:
            return 10
        lang = 'en-US'
        attributes = [('lang', lang)]
        self._xml_empty_tag('a:endParaRPr', attributes)

    def _write_a_r(self, title, font):
        if False:
            for i in range(10):
                print('nop')
        self._xml_start_tag('a:r')
        self._write_a_r_pr(font)
        self._write_a_t(title)
        self._xml_end_tag('a:r')

    def _write_a_r_pr(self, font):
        if False:
            while True:
                i = 10
        has_color = 0
        lang = 'en-US'
        style_attributes = Shape._get_font_style_attributes(font)
        latin_attributes = Shape._get_font_latin_attributes(font)
        if font and font['color'] is not None:
            has_color = 1
        style_attributes.insert(0, ('lang', lang))
        if latin_attributes or has_color:
            self._xml_start_tag('a:rPr', style_attributes)
            if has_color:
                self._write_a_solid_fill({'color': font['color']})
            if latin_attributes:
                self._write_a_latin(latin_attributes)
            self._xml_end_tag('a:rPr')
        else:
            self._xml_empty_tag('a:rPr', style_attributes)

    def _write_a_t(self, title):
        if False:
            while True:
                i = 10
        self._xml_data_element('a:t', title)

    def _write_tx_pr(self, font, is_y_axis=False):
        if False:
            while True:
                i = 10
        if font and font.get('rotation') is not None:
            rotation = font['rotation']
        else:
            rotation = None
        self._xml_start_tag('c:txPr')
        self._write_a_body_pr(rotation, is_y_axis)
        self._write_a_lst_style()
        self._write_a_p_formula(font)
        self._xml_end_tag('c:txPr')

    def _write_marker(self, marker):
        if False:
            return 10
        if marker is None:
            marker = self.default_marker
        if not marker:
            return
        if marker['type'] == 'automatic':
            return
        self._xml_start_tag('c:marker')
        self._write_symbol(marker['type'])
        if marker.get('size'):
            self._write_marker_size(marker['size'])
        self._write_sp_pr(marker)
        self._xml_end_tag('c:marker')

    def _write_marker_size(self, val):
        if False:
            print('Hello World!')
        attributes = [('val', val)]
        self._xml_empty_tag('c:size', attributes)

    def _write_symbol(self, val):
        if False:
            print('Hello World!')
        attributes = [('val', val)]
        self._xml_empty_tag('c:symbol', attributes)

    def _write_sp_pr(self, series):
        if False:
            while True:
                i = 10
        if not self._has_fill_formatting(series):
            return
        self._xml_start_tag('c:spPr')
        if series.get('fill') and series['fill']['defined']:
            if 'none' in series['fill']:
                self._write_a_no_fill()
            else:
                self._write_a_solid_fill(series['fill'])
        if series.get('pattern'):
            self._write_a_patt_fill(series['pattern'])
        if series.get('gradient'):
            self._write_a_grad_fill(series['gradient'])
        if series.get('line') and series['line']['defined']:
            self._write_a_ln(series['line'])
        self._xml_end_tag('c:spPr')

    def _write_a_ln(self, line):
        if False:
            while True:
                i = 10
        attributes = []
        width = line.get('width')
        if width is not None:
            width = int((width + 0.125) * 4) / 4.0
            width = int(0.5 + 12700 * width)
            attributes = [('w', width)]
        if line.get('none') or line.get('color') or line.get('dash_type'):
            self._xml_start_tag('a:ln', attributes)
            if 'none' in line:
                self._write_a_no_fill()
            elif 'color' in line:
                self._write_a_solid_fill(line)
            line_type = line.get('dash_type')
            if line_type:
                self._write_a_prst_dash(line_type)
            self._xml_end_tag('a:ln')
        else:
            self._xml_empty_tag('a:ln', attributes)

    def _write_a_no_fill(self):
        if False:
            return 10
        self._xml_empty_tag('a:noFill')

    def _write_a_solid_fill(self, fill):
        if False:
            for i in range(10):
                print('nop')
        self._xml_start_tag('a:solidFill')
        if 'color' in fill:
            color = get_rgb_color(fill['color'])
            transparency = fill.get('transparency')
            self._write_a_srgb_clr(color, transparency)
        self._xml_end_tag('a:solidFill')

    def _write_a_srgb_clr(self, val, transparency=None):
        if False:
            print('Hello World!')
        attributes = [('val', val)]
        if transparency:
            self._xml_start_tag('a:srgbClr', attributes)
            self._write_a_alpha(transparency)
            self._xml_end_tag('a:srgbClr')
        else:
            self._xml_empty_tag('a:srgbClr', attributes)

    def _write_a_alpha(self, val):
        if False:
            for i in range(10):
                print('nop')
        val = int((100 - int(val)) * 1000)
        attributes = [('val', val)]
        self._xml_empty_tag('a:alpha', attributes)

    def _write_a_prst_dash(self, val):
        if False:
            i = 10
            return i + 15
        attributes = [('val', val)]
        self._xml_empty_tag('a:prstDash', attributes)

    def _write_trendline(self, trendline):
        if False:
            for i in range(10):
                print('nop')
        if not trendline:
            return
        self._xml_start_tag('c:trendline')
        self._write_name(trendline.get('name'))
        self._write_sp_pr(trendline)
        self._write_trendline_type(trendline['type'])
        if trendline['type'] == 'poly':
            self._write_trendline_order(trendline.get('order'))
        if trendline['type'] == 'movingAvg':
            self._write_period(trendline.get('period'))
        self._write_forward(trendline.get('forward'))
        self._write_backward(trendline.get('backward'))
        if 'intercept' in trendline:
            self._write_c_intercept(trendline['intercept'])
        if trendline.get('display_r_squared'):
            self._write_c_disp_rsqr()
        if trendline.get('display_equation'):
            self._write_c_disp_eq()
            self._write_c_trendline_lbl(trendline)
        self._xml_end_tag('c:trendline')

    def _write_trendline_type(self, val):
        if False:
            for i in range(10):
                print('nop')
        attributes = [('val', val)]
        self._xml_empty_tag('c:trendlineType', attributes)

    def _write_name(self, data):
        if False:
            print('Hello World!')
        if data is None:
            return
        self._xml_data_element('c:name', data)

    def _write_trendline_order(self, val):
        if False:
            while True:
                i = 10
        if val < 2:
            val = 2
        attributes = [('val', val)]
        self._xml_empty_tag('c:order', attributes)

    def _write_period(self, val):
        if False:
            print('Hello World!')
        if val < 2:
            val = 2
        attributes = [('val', val)]
        self._xml_empty_tag('c:period', attributes)

    def _write_forward(self, val):
        if False:
            for i in range(10):
                print('nop')
        if not val:
            return
        attributes = [('val', val)]
        self._xml_empty_tag('c:forward', attributes)

    def _write_backward(self, val):
        if False:
            return 10
        if not val:
            return
        attributes = [('val', val)]
        self._xml_empty_tag('c:backward', attributes)

    def _write_c_intercept(self, val):
        if False:
            i = 10
            return i + 15
        attributes = [('val', val)]
        self._xml_empty_tag('c:intercept', attributes)

    def _write_c_disp_eq(self):
        if False:
            for i in range(10):
                print('nop')
        attributes = [('val', 1)]
        self._xml_empty_tag('c:dispEq', attributes)

    def _write_c_disp_rsqr(self):
        if False:
            return 10
        attributes = [('val', 1)]
        self._xml_empty_tag('c:dispRSqr', attributes)

    def _write_c_trendline_lbl(self, trendline):
        if False:
            for i in range(10):
                print('nop')
        self._xml_start_tag('c:trendlineLbl')
        self._write_layout(None, None)
        self._write_trendline_num_fmt()
        self._write_sp_pr(trendline['label'])
        if trendline['label']:
            font = trendline['label'].get('font')
            if font:
                self._write_axis_font(font)
        self._xml_end_tag('c:trendlineLbl')

    def _write_trendline_num_fmt(self):
        if False:
            return 10
        attributes = [('formatCode', 'General'), ('sourceLinked', 0)]
        self._xml_empty_tag('c:numFmt', attributes)

    def _write_hi_low_lines(self):
        if False:
            for i in range(10):
                print('nop')
        hi_low_lines = self.hi_low_lines
        if hi_low_lines is None:
            return
        if 'line' in hi_low_lines and hi_low_lines['line']['defined']:
            self._xml_start_tag('c:hiLowLines')
            self._write_sp_pr(hi_low_lines)
            self._xml_end_tag('c:hiLowLines')
        else:
            self._xml_empty_tag('c:hiLowLines')

    def _write_drop_lines(self):
        if False:
            print('Hello World!')
        drop_lines = self.drop_lines
        if drop_lines is None:
            return
        if drop_lines['line']['defined']:
            self._xml_start_tag('c:dropLines')
            self._write_sp_pr(drop_lines)
            self._xml_end_tag('c:dropLines')
        else:
            self._xml_empty_tag('c:dropLines')

    def _write_overlap(self, val):
        if False:
            i = 10
            return i + 15
        if val is None:
            return
        attributes = [('val', val)]
        self._xml_empty_tag('c:overlap', attributes)

    def _write_num_cache(self, data):
        if False:
            i = 10
            return i + 15
        if data:
            count = len(data)
        else:
            count = 0
        self._xml_start_tag('c:numCache')
        self._write_format_code('General')
        self._write_pt_count(count)
        for i in range(count):
            token = data[i]
            if token is None:
                continue
            try:
                float(token)
            except ValueError:
                token = 0
            self._write_pt(i, token)
        self._xml_end_tag('c:numCache')

    def _write_str_cache(self, data):
        if False:
            print('Hello World!')
        count = len(data)
        self._xml_start_tag('c:strCache')
        self._write_pt_count(count)
        for i in range(count):
            self._write_pt(i, data[i])
        self._xml_end_tag('c:strCache')

    def _write_format_code(self, data):
        if False:
            for i in range(10):
                print('nop')
        self._xml_data_element('c:formatCode', data)

    def _write_pt_count(self, val):
        if False:
            for i in range(10):
                print('nop')
        attributes = [('val', val)]
        self._xml_empty_tag('c:ptCount', attributes)

    def _write_pt(self, idx, value):
        if False:
            i = 10
            return i + 15
        if value is None:
            return
        attributes = [('idx', idx)]
        self._xml_start_tag('c:pt', attributes)
        self._write_v(value)
        self._xml_end_tag('c:pt')

    def _write_v(self, data):
        if False:
            while True:
                i = 10
        self._xml_data_element('c:v', data)

    def _write_protection(self):
        if False:
            print('Hello World!')
        if not self.protection:
            return
        self._xml_empty_tag('c:protection')

    def _write_d_pt(self, points):
        if False:
            print('Hello World!')
        index = -1
        if not points:
            return
        for point in points:
            index += 1
            if not point:
                continue
            self._write_d_pt_point(index, point)

    def _write_d_pt_point(self, index, point):
        if False:
            while True:
                i = 10
        self._xml_start_tag('c:dPt')
        self._write_idx(index)
        self._write_sp_pr(point)
        self._xml_end_tag('c:dPt')

    def _write_d_lbls(self, labels):
        if False:
            while True:
                i = 10
        if not labels:
            return
        self._xml_start_tag('c:dLbls')
        if labels.get('custom'):
            self._write_custom_labels(labels, labels['custom'])
        if labels.get('num_format'):
            self._write_data_label_number_format(labels['num_format'])
        self._write_sp_pr(labels)
        if labels.get('font'):
            self._write_axis_font(labels['font'])
        if labels.get('position'):
            self._write_d_lbl_pos(labels['position'])
        if labels.get('legend_key'):
            self._write_show_legend_key()
        if labels.get('value'):
            self._write_show_val()
        if labels.get('category'):
            self._write_show_cat_name()
        if labels.get('series_name'):
            self._write_show_ser_name()
        if labels.get('percentage'):
            self._write_show_percent()
        if labels.get('separator'):
            self._write_separator(labels['separator'])
        if labels.get('leader_lines'):
            self._write_show_leader_lines()
        self._xml_end_tag('c:dLbls')

    def _write_custom_labels(self, parent, labels):
        if False:
            print('Hello World!')
        index = 0
        for label in labels:
            index += 1
            if label is None:
                continue
            self._xml_start_tag('c:dLbl')
            self._write_idx(index - 1)
            delete_label = label.get('delete')
            if delete_label:
                self._write_delete(1)
            elif label.get('formula'):
                self._write_custom_label_formula(label)
                if parent.get('position'):
                    self._write_d_lbl_pos(parent['position'])
                if parent.get('value'):
                    self._write_show_val()
                if parent.get('category'):
                    self._write_show_cat_name()
                if parent.get('series_name'):
                    self._write_show_ser_name()
            elif label.get('value'):
                self._write_custom_label_str(label)
                if parent.get('position'):
                    self._write_d_lbl_pos(parent['position'])
                if parent.get('value'):
                    self._write_show_val()
                if parent.get('category'):
                    self._write_show_cat_name()
                if parent.get('series_name'):
                    self._write_show_ser_name()
            else:
                self._write_custom_label_format_only(label)
            self._xml_end_tag('c:dLbl')

    def _write_custom_label_str(self, label):
        if False:
            for i in range(10):
                print('nop')
        title = label.get('value')
        font = label.get('font')
        has_formatting = self._has_fill_formatting(label)
        self._write_layout(None, None)
        self._xml_start_tag('c:tx')
        self._write_rich(title, font, False, not has_formatting)
        self._xml_end_tag('c:tx')
        self._write_sp_pr(label)

    def _write_custom_label_formula(self, label):
        if False:
            while True:
                i = 10
        formula = label.get('formula')
        data_id = label.get('data_id')
        data = None
        if data_id is not None:
            data = self.formula_data[data_id]
        self._write_layout(None, None)
        self._xml_start_tag('c:tx')
        self._write_str_ref(formula, data, 'str')
        self._xml_end_tag('c:tx')
        self._write_custom_label_format_only(label)

    def _write_custom_label_format_only(self, label):
        if False:
            i = 10
            return i + 15
        font = label.get('font')
        has_formatting = self._has_fill_formatting(label)
        if has_formatting:
            self._write_sp_pr(label)
            self._write_tx_pr(font)
        elif font:
            self._xml_empty_tag('c:spPr')
            self._write_tx_pr(font)

    def _write_show_legend_key(self):
        if False:
            for i in range(10):
                print('nop')
        val = '1'
        attributes = [('val', val)]
        self._xml_empty_tag('c:showLegendKey', attributes)

    def _write_show_val(self):
        if False:
            return 10
        val = 1
        attributes = [('val', val)]
        self._xml_empty_tag('c:showVal', attributes)

    def _write_show_cat_name(self):
        if False:
            return 10
        val = 1
        attributes = [('val', val)]
        self._xml_empty_tag('c:showCatName', attributes)

    def _write_show_ser_name(self):
        if False:
            for i in range(10):
                print('nop')
        val = 1
        attributes = [('val', val)]
        self._xml_empty_tag('c:showSerName', attributes)

    def _write_show_percent(self):
        if False:
            print('Hello World!')
        val = 1
        attributes = [('val', val)]
        self._xml_empty_tag('c:showPercent', attributes)

    def _write_separator(self, data):
        if False:
            return 10
        self._xml_data_element('c:separator', data)

    def _write_show_leader_lines(self):
        if False:
            i = 10
            return i + 15
        uri = '{CE6537A1-D6FC-4f65-9D91-7224C49458BB}'
        xmlns_c_15 = 'http://schemas.microsoft.com/office/drawing/2012/chart'
        attributes = [('uri', uri), ('xmlns:c15', xmlns_c_15)]
        self._xml_start_tag('c:extLst')
        self._xml_start_tag('c:ext', attributes)
        self._xml_empty_tag('c15:showLeaderLines', [('val', 1)])
        self._xml_end_tag('c:ext')
        self._xml_end_tag('c:extLst')

    def _write_d_lbl_pos(self, val):
        if False:
            return 10
        attributes = [('val', val)]
        self._xml_empty_tag('c:dLblPos', attributes)

    def _write_delete(self, val):
        if False:
            while True:
                i = 10
        attributes = [('val', val)]
        self._xml_empty_tag('c:delete', attributes)

    def _write_c_invert_if_negative(self, invert):
        if False:
            return 10
        val = 1
        if not invert:
            return
        attributes = [('val', val)]
        self._xml_empty_tag('c:invertIfNegative', attributes)

    def _write_axis_font(self, font):
        if False:
            while True:
                i = 10
        if not font:
            return
        self._xml_start_tag('c:txPr')
        self._write_a_body_pr(font.get('rotation'), None)
        self._write_a_lst_style()
        self._xml_start_tag('a:p')
        self._write_a_p_pr_rich(font)
        self._write_a_end_para_rpr()
        self._xml_end_tag('a:p')
        self._xml_end_tag('c:txPr')

    def _write_a_latin(self, attributes):
        if False:
            i = 10
            return i + 15
        self._xml_empty_tag('a:latin', attributes)

    def _write_d_table(self):
        if False:
            i = 10
            return i + 15
        table = self.table
        if not table:
            return
        self._xml_start_tag('c:dTable')
        if table['horizontal']:
            self._write_show_horz_border()
        if table['vertical']:
            self._write_show_vert_border()
        if table['outline']:
            self._write_show_outline()
        if table['show_keys']:
            self._write_show_keys()
        if table['font']:
            self._write_tx_pr(table['font'])
        self._xml_end_tag('c:dTable')

    def _write_show_horz_border(self):
        if False:
            i = 10
            return i + 15
        attributes = [('val', 1)]
        self._xml_empty_tag('c:showHorzBorder', attributes)

    def _write_show_vert_border(self):
        if False:
            for i in range(10):
                print('nop')
        attributes = [('val', 1)]
        self._xml_empty_tag('c:showVertBorder', attributes)

    def _write_show_outline(self):
        if False:
            return 10
        attributes = [('val', 1)]
        self._xml_empty_tag('c:showOutline', attributes)

    def _write_show_keys(self):
        if False:
            return 10
        attributes = [('val', 1)]
        self._xml_empty_tag('c:showKeys', attributes)

    def _write_error_bars(self, error_bars):
        if False:
            while True:
                i = 10
        if not error_bars:
            return
        if error_bars['x_error_bars']:
            self._write_err_bars('x', error_bars['x_error_bars'])
        if error_bars['y_error_bars']:
            self._write_err_bars('y', error_bars['y_error_bars'])

    def _write_err_bars(self, direction, error_bars):
        if False:
            while True:
                i = 10
        if not error_bars:
            return
        self._xml_start_tag('c:errBars')
        self._write_err_dir(direction)
        self._write_err_bar_type(error_bars['direction'])
        self._write_err_val_type(error_bars['type'])
        if not error_bars['endcap']:
            self._write_no_end_cap()
        if error_bars['type'] == 'stdErr':
            pass
        elif error_bars['type'] == 'cust':
            self._write_custom_error(error_bars)
        else:
            self._write_error_val(error_bars['value'])
        self._write_sp_pr(error_bars)
        self._xml_end_tag('c:errBars')

    def _write_err_dir(self, val):
        if False:
            for i in range(10):
                print('nop')
        attributes = [('val', val)]
        self._xml_empty_tag('c:errDir', attributes)

    def _write_err_bar_type(self, val):
        if False:
            for i in range(10):
                print('nop')
        attributes = [('val', val)]
        self._xml_empty_tag('c:errBarType', attributes)

    def _write_err_val_type(self, val):
        if False:
            while True:
                i = 10
        attributes = [('val', val)]
        self._xml_empty_tag('c:errValType', attributes)

    def _write_no_end_cap(self):
        if False:
            i = 10
            return i + 15
        attributes = [('val', 1)]
        self._xml_empty_tag('c:noEndCap', attributes)

    def _write_error_val(self, val):
        if False:
            i = 10
            return i + 15
        attributes = [('val', val)]
        self._xml_empty_tag('c:val', attributes)

    def _write_custom_error(self, error_bars):
        if False:
            print('Hello World!')
        if error_bars['plus_values']:
            self._xml_start_tag('c:plus')
            if isinstance(error_bars['plus_values'], list):
                self._write_num_lit(error_bars['plus_values'])
            else:
                self._write_num_ref(error_bars['plus_values'], error_bars['plus_data'], 'num')
            self._xml_end_tag('c:plus')
        if error_bars['minus_values']:
            self._xml_start_tag('c:minus')
            if isinstance(error_bars['minus_values'], list):
                self._write_num_lit(error_bars['minus_values'])
            else:
                self._write_num_ref(error_bars['minus_values'], error_bars['minus_data'], 'num')
            self._xml_end_tag('c:minus')

    def _write_num_lit(self, data):
        if False:
            for i in range(10):
                print('nop')
        count = len(data)
        self._xml_start_tag('c:numLit')
        self._write_format_code('General')
        self._write_pt_count(count)
        for i in range(count):
            token = data[i]
            if token is None:
                continue
            try:
                float(token)
            except ValueError:
                token = 0
            self._write_pt(i, token)
        self._xml_end_tag('c:numLit')

    def _write_up_down_bars(self):
        if False:
            print('Hello World!')
        up_down_bars = self.up_down_bars
        if up_down_bars is None:
            return
        self._xml_start_tag('c:upDownBars')
        self._write_gap_width(150)
        self._write_up_bars(up_down_bars.get('up'))
        self._write_down_bars(up_down_bars.get('down'))
        self._xml_end_tag('c:upDownBars')

    def _write_gap_width(self, val):
        if False:
            while True:
                i = 10
        if val is None:
            return
        attributes = [('val', val)]
        self._xml_empty_tag('c:gapWidth', attributes)

    def _write_up_bars(self, bar_format):
        if False:
            i = 10
            return i + 15
        if bar_format['line'] and bar_format['line']['defined']:
            self._xml_start_tag('c:upBars')
            self._write_sp_pr(bar_format)
            self._xml_end_tag('c:upBars')
        else:
            self._xml_empty_tag('c:upBars')

    def _write_down_bars(self, bar_format):
        if False:
            print('Hello World!')
        if bar_format['line'] and bar_format['line']['defined']:
            self._xml_start_tag('c:downBars')
            self._write_sp_pr(bar_format)
            self._xml_end_tag('c:downBars')
        else:
            self._xml_empty_tag('c:downBars')

    def _write_disp_units(self, units, display):
        if False:
            return 10
        if not units:
            return
        attributes = [('val', units)]
        self._xml_start_tag('c:dispUnits')
        self._xml_empty_tag('c:builtInUnit', attributes)
        if display:
            self._xml_start_tag('c:dispUnitsLbl')
            self._xml_empty_tag('c:layout')
            self._xml_end_tag('c:dispUnitsLbl')
        self._xml_end_tag('c:dispUnits')

    def _write_a_grad_fill(self, gradient):
        if False:
            return 10
        attributes = [('flip', 'none'), ('rotWithShape', '1')]
        if gradient['type'] == 'linear':
            attributes = []
        self._xml_start_tag('a:gradFill', attributes)
        self._write_a_gs_lst(gradient)
        if gradient['type'] == 'linear':
            self._write_a_lin(gradient['angle'])
        else:
            self._write_a_path(gradient['type'])
            self._write_a_tile_rect(gradient['type'])
        self._xml_end_tag('a:gradFill')

    def _write_a_gs_lst(self, gradient):
        if False:
            i = 10
            return i + 15
        positions = gradient['positions']
        colors = gradient['colors']
        self._xml_start_tag('a:gsLst')
        for i in range(len(colors)):
            pos = int(positions[i] * 1000)
            attributes = [('pos', pos)]
            self._xml_start_tag('a:gs', attributes)
            color = get_rgb_color(colors[i])
            self._write_a_srgb_clr(color)
            self._xml_end_tag('a:gs')
        self._xml_end_tag('a:gsLst')

    def _write_a_lin(self, angle):
        if False:
            print('Hello World!')
        angle = int(60000 * angle)
        attributes = [('ang', angle), ('scaled', '0')]
        self._xml_empty_tag('a:lin', attributes)

    def _write_a_path(self, gradient_type):
        if False:
            i = 10
            return i + 15
        attributes = [('path', gradient_type)]
        self._xml_start_tag('a:path', attributes)
        self._write_a_fill_to_rect(gradient_type)
        self._xml_end_tag('a:path')

    def _write_a_fill_to_rect(self, gradient_type):
        if False:
            while True:
                i = 10
        if gradient_type == 'shape':
            attributes = [('l', '50000'), ('t', '50000'), ('r', '50000'), ('b', '50000')]
        else:
            attributes = [('l', '100000'), ('t', '100000')]
        self._xml_empty_tag('a:fillToRect', attributes)

    def _write_a_tile_rect(self, gradient_type):
        if False:
            return 10
        if gradient_type == 'shape':
            attributes = []
        else:
            attributes = [('r', '-100000'), ('b', '-100000')]
        self._xml_empty_tag('a:tileRect', attributes)

    def _write_a_patt_fill(self, pattern):
        if False:
            i = 10
            return i + 15
        attributes = [('prst', pattern['pattern'])]
        self._xml_start_tag('a:pattFill', attributes)
        self._write_a_fg_clr(pattern['fg_color'])
        self._write_a_bg_clr(pattern['bg_color'])
        self._xml_end_tag('a:pattFill')

    def _write_a_fg_clr(self, color):
        if False:
            while True:
                i = 10
        color = get_rgb_color(color)
        self._xml_start_tag('a:fgClr')
        self._write_a_srgb_clr(color)
        self._xml_end_tag('a:fgClr')

    def _write_a_bg_clr(self, color):
        if False:
            return 10
        color = get_rgb_color(color)
        self._xml_start_tag('a:bgClr')
        self._write_a_srgb_clr(color)
        self._xml_end_tag('a:bgClr')