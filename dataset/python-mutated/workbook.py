import hashlib
import operator
import os
import re
import time
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from struct import unpack
from warnings import warn
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED, LargeZipFile
from . import xmlwriter
from .worksheet import Worksheet
from .chartsheet import Chartsheet
from .sharedstrings import SharedStringTable
from .format import Format
from .packager import Packager
from .utility import xl_cell_to_rowcol
from .chart_area import ChartArea
from .chart_bar import ChartBar
from .chart_column import ChartColumn
from .chart_doughnut import ChartDoughnut
from .chart_line import ChartLine
from .chart_pie import ChartPie
from .chart_radar import ChartRadar
from .chart_scatter import ChartScatter
from .chart_stock import ChartStock
from .exceptions import InvalidWorksheetName
from .exceptions import DuplicateWorksheetName
from .exceptions import UndefinedImageSize
from .exceptions import UnsupportedImageFormat
from .exceptions import FileCreateError
from .exceptions import FileSizeError

class Workbook(xmlwriter.XMLwriter):
    """
    A class for writing the Excel XLSX Workbook file.


    """
    chartsheet_class = Chartsheet
    worksheet_class = Worksheet

    def __init__(self, filename=None, options=None):
        if False:
            print('Hello World!')
        '\n        Constructor.\n\n        '
        if options is None:
            options = {}
        super(Workbook, self).__init__()
        self.filename = filename
        self.tmpdir = options.get('tmpdir', None)
        self.date_1904 = options.get('date_1904', False)
        self.strings_to_numbers = options.get('strings_to_numbers', False)
        self.strings_to_formulas = options.get('strings_to_formulas', True)
        self.strings_to_urls = options.get('strings_to_urls', True)
        self.nan_inf_to_errors = options.get('nan_inf_to_errors', False)
        self.default_date_format = options.get('default_date_format', None)
        self.constant_memory = options.get('constant_memory', False)
        self.in_memory = options.get('in_memory', False)
        self.excel2003_style = options.get('excel2003_style', False)
        self.remove_timezone = options.get('remove_timezone', False)
        self.use_future_functions = options.get('use_future_functions', False)
        self.default_format_properties = options.get('default_format_properties', {})
        self.max_url_length = options.get('max_url_length', 2079)
        if self.max_url_length < 255:
            self.max_url_length = 2079
        if options.get('use_zip64'):
            self.allow_zip64 = True
        else:
            self.allow_zip64 = False
        self.worksheet_meta = WorksheetMeta()
        self.selected = 0
        self.fileclosed = 0
        self.filehandle = None
        self.internal_fh = 0
        self.sheet_name = 'Sheet'
        self.chart_name = 'Chart'
        self.sheetname_count = 0
        self.chartname_count = 0
        self.worksheets_objs = []
        self.charts = []
        self.drawings = []
        self.sheetnames = {}
        self.formats = []
        self.xf_formats = []
        self.xf_format_indices = {}
        self.dxf_formats = []
        self.dxf_format_indices = {}
        self.palette = []
        self.font_count = 0
        self.num_formats = []
        self.defined_names = []
        self.named_ranges = []
        self.custom_colors = []
        self.doc_properties = {}
        self.custom_properties = []
        self.createtime = datetime.now(timezone.utc)
        self.num_vml_files = 0
        self.num_comment_files = 0
        self.x_window = 240
        self.y_window = 15
        self.window_width = 16095
        self.window_height = 9660
        self.tab_ratio = 600
        self.str_table = SharedStringTable()
        self.vba_project = None
        self.vba_project_is_stream = False
        self.vba_project_signature = None
        self.vba_project_signature_is_stream = False
        self.vba_codename = None
        self.image_types = {}
        self.images = []
        self.border_count = 0
        self.fill_count = 0
        self.drawing_count = 0
        self.calc_mode = 'auto'
        self.calc_on_load = True
        self.calc_id = 124519
        self.has_comments = False
        self.read_only = 0
        self.has_metadata = False
        if self.in_memory:
            self.constant_memory = False
        if self.excel2003_style:
            self.add_format({'xf_index': 0, 'font_family': 0})
        else:
            self.add_format({'xf_index': 0})
        self.default_url_format = self.add_format({'hyperlink': True})
        if self.default_date_format is not None:
            self.default_date_format = self.add_format({'num_format': self.default_date_format})

    def __enter__(self):
        if False:
            return 10
        'Return self object to use with "with" statement.'
        return self

    def __exit__(self, type, value, traceback):
        if False:
            while True:
                i = 10
        'Close workbook when exiting "with" statement.'
        self.close()

    def add_worksheet(self, name=None, worksheet_class=None):
        if False:
            while True:
                i = 10
        "\n        Add a new worksheet to the Excel workbook.\n\n        Args:\n            name: The worksheet name. Defaults to 'Sheet1', etc.\n\n        Returns:\n            Reference to a worksheet object.\n\n        "
        if worksheet_class is None:
            worksheet_class = self.worksheet_class
        return self._add_sheet(name, worksheet_class=worksheet_class)

    def add_chartsheet(self, name=None, chartsheet_class=None):
        if False:
            return 10
        "\n        Add a new chartsheet to the Excel workbook.\n\n        Args:\n            name: The chartsheet name. Defaults to 'Sheet1', etc.\n\n        Returns:\n            Reference to a chartsheet object.\n\n        "
        if chartsheet_class is None:
            chartsheet_class = self.chartsheet_class
        return self._add_sheet(name, worksheet_class=chartsheet_class)

    def add_format(self, properties=None):
        if False:
            return 10
        '\n        Add a new Format to the Excel Workbook.\n\n        Args:\n            properties: The format properties.\n\n        Returns:\n            Reference to a Format object.\n\n        '
        format_properties = self.default_format_properties.copy()
        if self.excel2003_style:
            format_properties = {'font_name': 'Arial', 'font_size': 10, 'theme': 1 * -1}
        if properties:
            format_properties.update(properties)
        xf_format = Format(format_properties, self.xf_format_indices, self.dxf_format_indices)
        self.formats.append(xf_format)
        return xf_format

    def add_chart(self, options):
        if False:
            return 10
        '\n        Create a chart object.\n\n        Args:\n            options: The chart type and subtype options.\n\n        Returns:\n            Reference to a Chart object.\n\n        '
        chart_type = options.get('type')
        if chart_type is None:
            warn('Chart type must be defined in add_chart()')
            return
        if chart_type == 'area':
            chart = ChartArea(options)
        elif chart_type == 'bar':
            chart = ChartBar(options)
        elif chart_type == 'column':
            chart = ChartColumn(options)
        elif chart_type == 'doughnut':
            chart = ChartDoughnut(options)
        elif chart_type == 'line':
            chart = ChartLine(options)
        elif chart_type == 'pie':
            chart = ChartPie(options)
        elif chart_type == 'radar':
            chart = ChartRadar(options)
        elif chart_type == 'scatter':
            chart = ChartScatter(options)
        elif chart_type == 'stock':
            chart = ChartStock(options)
        else:
            warn("Unknown chart type '%s' in add_chart()" % chart_type)
            return
        if 'name' in options:
            chart.chart_name = options['name']
        chart.embedded = True
        chart.date_1904 = self.date_1904
        chart.remove_timezone = self.remove_timezone
        self.charts.append(chart)
        return chart

    def add_vba_project(self, vba_project, is_stream=False):
        if False:
            return 10
        '\n        Add a vbaProject binary to the Excel workbook.\n\n        Args:\n            vba_project: The vbaProject binary file name.\n            is_stream:   vba_project is an in memory byte stream.\n\n        Returns:\n            Nothing.\n\n        '
        if not is_stream and (not os.path.exists(vba_project)):
            warn("VBA project binary file '%s' not found." % vba_project)
            return -1
        if self.vba_codename is None:
            self.vba_codename = 'ThisWorkbook'
        self.vba_project = vba_project
        self.vba_project_is_stream = is_stream

    def add_signed_vba_project(self, vba_project, signature, project_is_stream=False, signature_is_stream=False):
        if False:
            while True:
                i = 10
        '\n        Add a vbaProject binary and a vbaProjectSignature binary to the\n        Excel workbook.\n\n        Args:\n            vba_project:           The vbaProject binary file name.\n            signature:             The vbaProjectSignature binary file name.\n            project_is_stream:     vba_project is an in memory byte stream.\n            signature_is_stream:   signature is an in memory byte stream.\n\n        Returns:\n            Nothing.\n\n        '
        if self.add_vba_project(vba_project, project_is_stream) == -1:
            return -1
        if not signature_is_stream and (not os.path.exists(signature)):
            warn("VBA project signature binary file '%s' not found." % signature)
            return -1
        self.vba_project_signature = signature
        self.vba_project_signature_is_stream = signature_is_stream

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Call finalization code and close file.\n\n        Args:\n            None.\n\n        Returns:\n            Nothing.\n\n        '
        if not self.fileclosed:
            try:
                self._store_workbook()
            except IOError as e:
                raise FileCreateError(e)
            except LargeZipFile:
                raise FileSizeError('Filesize would require ZIP64 extensions. Use workbook.use_zip64().')
            self.fileclosed = True
            if self.constant_memory:
                for worksheet in self.worksheets():
                    worksheet._opt_close()
        else:
            warn('Calling close() on already closed file.')

    def set_size(self, width, height):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the size of a workbook window.\n\n        Args:\n            width:  Width  of the window in pixels.\n            height: Height of the window in pixels.\n\n        Returns:\n            Nothing.\n\n        '
        if width:
            self.window_width = int(width * 1440 / 96)
        else:
            self.window_width = 16095
        if height:
            self.window_height = int(height * 1440 / 96)
        else:
            self.window_height = 9660

    def set_tab_ratio(self, tab_ratio=None):
        if False:
            print('Hello World!')
        '\n        Set the ratio between worksheet tabs and the horizontal slider.\n\n        Args:\n            tab_ratio: The tab ratio, 0 <= tab_ratio <= 100\n\n        Returns:\n            Nothing.\n\n        '
        if tab_ratio is None:
            return
        if tab_ratio < 0 or tab_ratio > 100:
            warn("Tab ratio '%d' outside: 0 <= tab_ratio <= 100" % tab_ratio)
        else:
            self.tab_ratio = int(tab_ratio * 10)

    def set_properties(self, properties):
        if False:
            print('Hello World!')
        '\n        Set the document properties such as Title, Author etc.\n\n        Args:\n            properties: Dictionary of document properties.\n\n        Returns:\n            Nothing.\n\n        '
        self.doc_properties = properties

    def set_custom_property(self, name, value, property_type=None):
        if False:
            return 10
        '\n        Set a custom document property.\n\n        Args:\n            name:          The name of the custom property.\n            value:         The value of the custom property.\n            property_type: The type of the custom property. Optional.\n\n        Returns:\n            Nothing.\n\n        '
        if name is None or value is None:
            warn('The name and value parameters must be non-None in set_custom_property()')
            return -1
        if property_type is None:
            if isinstance(value, bool):
                property_type = 'bool'
            elif isinstance(value, datetime):
                property_type = 'date'
            elif isinstance(value, int):
                property_type = 'number_int'
            elif isinstance(value, (float, int, Decimal, Fraction)):
                property_type = 'number'
            else:
                property_type = 'text'
        if property_type == 'date':
            value = value.strftime('%Y-%m-%dT%H:%M:%SZ')
        if property_type == 'text' and len(value) > 255:
            warn("Length of 'value' parameter exceeds Excel's limit of 255 characters in set_custom_property(): '%s'" % value)
        if len(name) > 255:
            warn("Length of 'name' parameter exceeds Excel's limit of 255 characters in set_custom_property(): '%s'" % name)
        self.custom_properties.append((name, value, property_type))

    def set_calc_mode(self, mode, calc_id=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the Excel calculation mode for the workbook.\n\n        Args:\n            mode: String containing one of:\n                * manual\n                * auto_except_tables\n                * auto\n\n        Returns:\n            Nothing.\n\n        '
        self.calc_mode = mode
        if mode == 'manual':
            self.calc_on_load = False
        elif mode == 'auto_except_tables':
            self.calc_mode = 'autoNoTable'
        if calc_id:
            self.calc_id = calc_id

    def define_name(self, name, formula):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a defined name in the workbook.\n\n        Args:\n            name:    The defined name.\n            formula: The cell or range that the defined name refers to.\n\n        Returns:\n            Nothing.\n\n        '
        sheet_index = None
        sheetname = ''
        if formula.startswith('='):
            formula = formula.lstrip('=')
        sheet_parts = re.compile('^([^!]+)!([^!]+)$')
        match = sheet_parts.match(name)
        if match:
            sheetname = match.group(1)
            name = match.group(2)
            sheet_index = self._get_sheet_index(sheetname)
            if sheet_index is None:
                warn("Unknown sheet name '%s' in defined_name()" % sheetname)
                return -1
        else:
            sheet_index = -1
        if not re.match('^[\\w\\\\][\\w\\\\.]*$', name, re.UNICODE) or re.match('^\\d', name):
            warn("Invalid Excel characters in defined_name(): '%s'" % name)
            return -1
        if re.match('^[a-zA-Z][a-zA-Z]?[a-dA-D]?\\d+$', name):
            warn("Name looks like a cell name in defined_name(): '%s'" % name)
            return -1
        if re.match('^[rcRC]$', name) or re.match('^[rcRC]\\d+[rcRC]\\d+$', name):
            warn("Invalid name '%s' like a RC cell ref in defined_name()" % name)
            return -1
        self.defined_names.append([name, sheet_index, formula, False])

    def worksheets(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a list of the worksheet objects in the workbook.\n\n        Args:\n            None.\n\n        Returns:\n            A list of worksheet objects.\n\n        '
        return self.worksheets_objs

    def get_worksheet_by_name(self, name):
        if False:
            print('Hello World!')
        '\n        Return a worksheet object in the workbook using the sheetname.\n\n        Args:\n            name: The name of the worksheet.\n\n        Returns:\n            A worksheet object or None.\n\n        '
        return self.sheetnames.get(name)

    def get_default_url_format(self):
        if False:
            while True:
                i = 10
        "\n        Get the default url format used when a user defined format isn't\n        specified with write_url(). The format is the hyperlink style defined\n        by Excel for the default theme.\n\n        Args:\n            None.\n\n        Returns:\n            A format object.\n\n        "
        return self.default_url_format

    def use_zip64(self):
        if False:
            print('Hello World!')
        '\n        Allow ZIP64 extensions when writing xlsx file zip container.\n\n        Args:\n            None.\n\n        Returns:\n            Nothing.\n\n        '
        self.allow_zip64 = True

    def set_vba_name(self, name=None):
        if False:
            print('Hello World!')
        '\n        Set the VBA name for the workbook. By default the workbook is referred\n        to as ThisWorkbook in VBA.\n\n        Args:\n            name: The VBA name for the workbook.\n\n        Returns:\n            Nothing.\n\n        '
        if name is not None:
            self.vba_codename = name
        else:
            self.vba_codename = 'ThisWorkbook'

    def read_only_recommended(self):
        if False:
            print('Hello World!')
        '\n        Set the Excel "Read-only recommended" option when saving a file.\n\n        Args:\n            None.\n\n        Returns:\n            Nothing.\n\n        '
        self.read_only = 2

    def _assemble_xml_file(self):
        if False:
            for i in range(10):
                print('nop')
        self._prepare_format_properties()
        self._xml_declaration()
        self._write_workbook()
        self._write_file_version()
        self._write_file_sharing()
        self._write_workbook_pr()
        self._write_book_views()
        self._write_sheets()
        self._write_defined_names()
        self._write_calc_pr()
        self._xml_end_tag('workbook')
        self._xml_close()

    def _store_workbook(self):
        if False:
            while True:
                i = 10
        try:
            xlsx_file = ZipFile(self.filename, 'w', compression=ZIP_DEFLATED, allowZip64=self.allow_zip64)
        except IOError as e:
            raise e
        packager = self._get_packager()
        if not self.worksheets():
            self.add_worksheet()
        if self.worksheet_meta.activesheet == 0:
            self.worksheets_objs[0].selected = 1
            self.worksheets_objs[0].hidden = 0
        for sheet in self.worksheets():
            if sheet.index == self.worksheet_meta.activesheet:
                sheet.active = 1
        if self.vba_project:
            for sheet in self.worksheets():
                if sheet.vba_codename is None:
                    sheet.set_vba_name()
        self._prepare_sst_string_data()
        self._prepare_vml()
        self._prepare_defined_names()
        self._prepare_drawings()
        self._add_chart_data()
        self._prepare_tables()
        self._prepare_metadata()
        packager._add_workbook(self)
        packager._set_tmpdir(self.tmpdir)
        packager._set_in_memory(self.in_memory)
        xml_files = packager._create_package()
        packager = None
        for (file_id, file_data) in enumerate(xml_files):
            (os_filename, xml_filename, is_binary) = file_data
            if self.in_memory:
                zipinfo = ZipInfo(xml_filename, (1980, 1, 1, 0, 0, 0))
                zipinfo.compress_type = xlsx_file.compression
                if is_binary:
                    xlsx_file.writestr(zipinfo, os_filename.getvalue())
                else:
                    xlsx_file.writestr(zipinfo, os_filename.getvalue().encode('utf-8'))
            else:
                timestamp = time.mktime((1980, 1, 31, 0, 0, 0, 0, 0, -1))
                os.utime(os_filename, (timestamp, timestamp))
                try:
                    xlsx_file.write(os_filename, xml_filename)
                    os.remove(os_filename)
                except LargeZipFile as e:
                    for i in range(file_id, len(xml_files) - 1):
                        os.remove(xml_files[i][0])
                    raise e
        xlsx_file.close()

    def _add_sheet(self, name, worksheet_class=None):
        if False:
            while True:
                i = 10
        if worksheet_class:
            worksheet = worksheet_class()
        else:
            worksheet = self.worksheet_class()
        sheet_index = len(self.worksheets_objs)
        name = self._check_sheetname(name, isinstance(worksheet, Chartsheet))
        init_data = {'name': name, 'index': sheet_index, 'str_table': self.str_table, 'worksheet_meta': self.worksheet_meta, 'constant_memory': self.constant_memory, 'tmpdir': self.tmpdir, 'date_1904': self.date_1904, 'strings_to_numbers': self.strings_to_numbers, 'strings_to_formulas': self.strings_to_formulas, 'strings_to_urls': self.strings_to_urls, 'nan_inf_to_errors': self.nan_inf_to_errors, 'default_date_format': self.default_date_format, 'default_url_format': self.default_url_format, 'excel2003_style': self.excel2003_style, 'remove_timezone': self.remove_timezone, 'max_url_length': self.max_url_length, 'use_future_functions': self.use_future_functions}
        worksheet._initialize(init_data)
        self.worksheets_objs.append(worksheet)
        self.sheetnames[name] = worksheet
        return worksheet

    def _check_sheetname(self, sheetname, is_chartsheet=False):
        if False:
            for i in range(10):
                print('nop')
        invalid_char = re.compile('[\\[\\]:*?/\\\\]')
        if is_chartsheet:
            self.chartname_count += 1
        else:
            self.sheetname_count += 1
        if sheetname is None or sheetname == '':
            if is_chartsheet:
                sheetname = self.chart_name + str(self.chartname_count)
            else:
                sheetname = self.sheet_name + str(self.sheetname_count)
        if len(sheetname) > 31:
            raise InvalidWorksheetName("Excel worksheet name '%s' must be <= 31 chars." % sheetname)
        if invalid_char.search(sheetname):
            raise InvalidWorksheetName("Invalid Excel character '[]:*?/\\' in sheetname '%s'." % sheetname)
        if sheetname.startswith("'") or sheetname.endswith("'"):
            raise InvalidWorksheetName('Sheet name cannot start or end with an apostrophe "%s".' % sheetname)
        for worksheet in self.worksheets():
            if sheetname.lower() == worksheet.name.lower():
                raise DuplicateWorksheetName("Sheetname '%s', with case ignored, is already in use." % sheetname)
        return sheetname

    def _prepare_format_properties(self):
        if False:
            while True:
                i = 10
        self._prepare_formats()
        self._prepare_fonts()
        self._prepare_num_formats()
        self._prepare_borders()
        self._prepare_fills()

    def _prepare_formats(self):
        if False:
            for i in range(10):
                print('nop')
        xf_formats = []
        dxf_formats = []
        for xf_format in self.formats:
            if xf_format.xf_index is not None:
                xf_formats.append(xf_format)
            if xf_format.dxf_index is not None:
                dxf_formats.append(xf_format)
        self.xf_formats = [None] * len(xf_formats)
        self.dxf_formats = [None] * len(dxf_formats)
        for xf_format in xf_formats:
            index = xf_format.xf_index
            self.xf_formats[index] = xf_format
        for dxf_format in dxf_formats:
            index = dxf_format.dxf_index
            self.dxf_formats[index] = dxf_format

    def _set_default_xf_indices(self):
        if False:
            i = 10
            return i + 15
        formats = list(self.formats)
        del formats[1]
        if self.default_date_format is not None:
            del formats[1]
        for xf_format in formats:
            xf_format._get_xf_index()

    def _prepare_fonts(self):
        if False:
            print('Hello World!')
        fonts = {}
        index = 0
        for xf_format in self.xf_formats:
            key = xf_format._get_font_key()
            if key in fonts:
                xf_format.font_index = fonts[key]
                xf_format.has_font = 0
            else:
                fonts[key] = index
                xf_format.font_index = index
                xf_format.has_font = 1
                index += 1
        self.font_count = index
        for xf_format in self.dxf_formats:
            if xf_format.font_color or xf_format.bold or xf_format.italic or xf_format.underline or xf_format.font_strikeout:
                xf_format.has_dxf_font = 1

    def _prepare_num_formats(self):
        if False:
            for i in range(10):
                print('nop')
        unique_num_formats = {}
        num_formats = []
        index = 164
        for xf_format in self.xf_formats + self.dxf_formats:
            num_format = xf_format.num_format
            if not isinstance(num_format, str):
                num_format = int(num_format)
                if num_format == 0:
                    num_format = 1
                xf_format.num_format_index = num_format
                continue
            elif num_format == '0':
                xf_format.num_format_index = 1
                continue
            elif num_format == 'General':
                xf_format.num_format_index = 0
                continue
            if num_format in unique_num_formats:
                xf_format.num_format_index = unique_num_formats[num_format]
            else:
                unique_num_formats[num_format] = index
                xf_format.num_format_index = index
                index += 1
                if xf_format.xf_index:
                    num_formats.append(num_format)
        self.num_formats = num_formats

    def _prepare_borders(self):
        if False:
            print('Hello World!')
        borders = {}
        index = 0
        for xf_format in self.xf_formats:
            key = xf_format._get_border_key()
            if key in borders:
                xf_format.border_index = borders[key]
                xf_format.has_border = 0
            else:
                borders[key] = index
                xf_format.border_index = index
                xf_format.has_border = 1
                index += 1
        self.border_count = index
        has_border = re.compile('[^0:]')
        for xf_format in self.dxf_formats:
            key = xf_format._get_border_key()
            if has_border.search(key):
                xf_format.has_dxf_border = 1

    def _prepare_fills(self):
        if False:
            while True:
                i = 10
        fills = {}
        index = 2
        fills['0:0:0'] = 0
        fills['17:0:0'] = 1
        for xf_format in self.dxf_formats:
            if xf_format.pattern or xf_format.bg_color or xf_format.fg_color:
                xf_format.has_dxf_fill = 1
                xf_format.dxf_bg_color = xf_format.bg_color
                xf_format.dxf_fg_color = xf_format.fg_color
        for xf_format in self.xf_formats:
            if xf_format.pattern == 1 and xf_format.bg_color != 0 and (xf_format.fg_color != 0):
                tmp = xf_format.fg_color
                xf_format.fg_color = xf_format.bg_color
                xf_format.bg_color = tmp
            if xf_format.pattern <= 1 and xf_format.bg_color != 0 and (xf_format.fg_color == 0):
                xf_format.fg_color = xf_format.bg_color
                xf_format.bg_color = 0
                xf_format.pattern = 1
            if xf_format.pattern <= 1 and xf_format.bg_color == 0 and (xf_format.fg_color != 0):
                xf_format.pattern = 1
            key = xf_format._get_fill_key()
            if key in fills:
                xf_format.fill_index = fills[key]
                xf_format.has_fill = 0
            else:
                fills[key] = index
                xf_format.fill_index = index
                xf_format.has_fill = 1
                index += 1
        self.fill_count = index

    def _prepare_defined_names(self):
        if False:
            return 10
        defined_names = self.defined_names
        for sheet in self.worksheets():
            if sheet.autofilter_area:
                hidden = 1
                sheet_range = sheet.autofilter_area
                defined_names.append(['_xlnm._FilterDatabase', sheet.index, sheet_range, hidden])
            if sheet.print_area_range:
                hidden = 0
                sheet_range = sheet.print_area_range
                defined_names.append(['_xlnm.Print_Area', sheet.index, sheet_range, hidden])
            if sheet.repeat_col_range or sheet.repeat_row_range:
                hidden = 0
                sheet_range = ''
                if sheet.repeat_col_range and sheet.repeat_row_range:
                    sheet_range = sheet.repeat_col_range + ',' + sheet.repeat_row_range
                else:
                    sheet_range = sheet.repeat_col_range + sheet.repeat_row_range
                defined_names.append(['_xlnm.Print_Titles', sheet.index, sheet_range, hidden])
        defined_names = self._sort_defined_names(defined_names)
        self.defined_names = defined_names
        self.named_ranges = self._extract_named_ranges(defined_names)

    def _sort_defined_names(self, names):
        if False:
            print('Hello World!')
        for name_list in names:
            (defined_name, _, sheet_name, _) = name_list
            defined_name = defined_name.replace('_xlnm.', '').lower()
            sheet_name = sheet_name.lstrip("'").lower()
            name_list.append(defined_name + '::' + sheet_name)
        names.sort(key=operator.itemgetter(4))
        for name_list in names:
            name_list.pop()
        return names

    def _prepare_drawings(self):
        if False:
            for i in range(10):
                print('nop')
        chart_ref_id = 0
        image_ref_id = 0
        ref_id = 0
        drawing_id = 0
        image_ids = {}
        header_image_ids = {}
        background_ids = {}
        for sheet in self.worksheets():
            chart_count = len(sheet.charts)
            image_count = len(sheet.images)
            shape_count = len(sheet.shapes)
            header_image_count = len(sheet.header_images)
            footer_image_count = len(sheet.footer_images)
            has_background = sheet.background_image
            has_drawing = False
            if not (chart_count or image_count or shape_count or header_image_count or footer_image_count or has_background):
                continue
            if chart_count or image_count or shape_count:
                drawing_id += 1
                has_drawing = True
            if sheet.background_image:
                if sheet.background_bytes:
                    filename = ''
                    image_data = sheet.background_image
                else:
                    filename = sheet.background_image
                    image_data = None
                (image_type, _, _, _, _, _, digest) = self._get_image_properties(filename, image_data)
                if digest in background_ids:
                    ref_id = background_ids[digest]
                else:
                    image_ref_id += 1
                    ref_id = image_ref_id
                    background_ids[digest] = image_ref_id
                    self.images.append([filename, image_type, image_data])
                sheet._prepare_background(ref_id, image_type)
            for index in range(image_count):
                filename = sheet.images[index][2]
                image_data = sheet.images[index][10]
                (image_type, width, height, name, x_dpi, y_dpi, digest) = self._get_image_properties(filename, image_data)
                if digest in image_ids:
                    ref_id = image_ids[digest]
                else:
                    image_ref_id += 1
                    ref_id = image_ref_id
                    image_ids[digest] = image_ref_id
                    self.images.append([filename, image_type, image_data])
                sheet._prepare_image(index, ref_id, drawing_id, width, height, name, image_type, x_dpi, y_dpi, digest)
            for index in range(chart_count):
                chart_ref_id += 1
                sheet._prepare_chart(index, chart_ref_id, drawing_id)
            for index in range(shape_count):
                sheet._prepare_shape(index, drawing_id)
            for index in range(header_image_count):
                filename = sheet.header_images[index][0]
                image_data = sheet.header_images[index][1]
                position = sheet.header_images[index][2]
                (image_type, width, height, name, x_dpi, y_dpi, digest) = self._get_image_properties(filename, image_data)
                if digest in header_image_ids:
                    ref_id = header_image_ids[digest]
                else:
                    image_ref_id += 1
                    ref_id = image_ref_id
                    header_image_ids[digest] = image_ref_id
                    self.images.append([filename, image_type, image_data])
                sheet._prepare_header_image(ref_id, width, height, name, image_type, position, x_dpi, y_dpi, digest)
            for index in range(footer_image_count):
                filename = sheet.footer_images[index][0]
                image_data = sheet.footer_images[index][1]
                position = sheet.footer_images[index][2]
                (image_type, width, height, name, x_dpi, y_dpi, digest) = self._get_image_properties(filename, image_data)
                if digest in header_image_ids:
                    ref_id = header_image_ids[digest]
                else:
                    image_ref_id += 1
                    ref_id = image_ref_id
                    header_image_ids[digest] = image_ref_id
                    self.images.append([filename, image_type, image_data])
                sheet._prepare_header_image(ref_id, width, height, name, image_type, position, x_dpi, y_dpi, digest)
            if has_drawing:
                drawing = sheet.drawing
                self.drawings.append(drawing)
        for chart in self.charts[:]:
            if chart.id == -1:
                self.charts.remove(chart)
        self.charts = sorted(self.charts, key=lambda chart: chart.id)
        self.drawing_count = drawing_id

    def _get_image_properties(self, filename, image_data):
        if False:
            while True:
                i = 10
        height = 0
        width = 0
        x_dpi = 96
        y_dpi = 96
        if not image_data:
            fh = open(filename, 'rb')
            data = fh.read()
        else:
            data = image_data.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        image_name = os.path.basename(filename)
        marker1 = unpack('3s', data[1:4])[0]
        marker2 = unpack('>H', data[:2])[0]
        marker3 = unpack('2s', data[:2])[0]
        marker4 = unpack('<L', data[:4])[0]
        marker5 = unpack('4s', data[40:44])[0]
        marker6 = unpack('4s', data[:4])[0]
        png_marker = b'PNG'
        bmp_marker = b'BM'
        emf_marker = b' EMF'
        gif_marker = b'GIF8'
        if marker1 == png_marker:
            self.image_types['png'] = True
            (image_type, width, height, x_dpi, y_dpi) = self._process_png(data)
        elif marker2 == 65496:
            self.image_types['jpeg'] = True
            (image_type, width, height, x_dpi, y_dpi) = self._process_jpg(data)
        elif marker3 == bmp_marker:
            self.image_types['bmp'] = True
            (image_type, width, height) = self._process_bmp(data)
        elif marker4 == 2596720087:
            self.image_types['wmf'] = True
            (image_type, width, height, x_dpi, y_dpi) = self._process_wmf(data)
        elif marker4 == 1 and marker5 == emf_marker:
            self.image_types['emf'] = True
            (image_type, width, height, x_dpi, y_dpi) = self._process_emf(data)
        elif marker6 == gif_marker:
            self.image_types['gif'] = True
            (image_type, width, height, x_dpi, y_dpi) = self._process_gif(data)
        else:
            raise UnsupportedImageFormat('%s: Unknown or unsupported image file format.' % filename)
        if not height or not width:
            raise UndefinedImageSize('%s: no size data found in image file.' % filename)
        if not image_data:
            fh.close()
        if x_dpi == 0:
            x_dpi = 96
        if y_dpi == 0:
            y_dpi = 96
        return (image_type, width, height, image_name, x_dpi, y_dpi, digest)

    def _process_png(self, data):
        if False:
            for i in range(10):
                print('nop')
        offset = 8
        data_length = len(data)
        end_marker = False
        width = 0
        height = 0
        x_dpi = 96
        y_dpi = 96
        while not end_marker and offset < data_length:
            length = unpack('>I', data[offset + 0:offset + 4])[0]
            marker = unpack('4s', data[offset + 4:offset + 8])[0]
            if marker == b'IHDR':
                width = unpack('>I', data[offset + 8:offset + 12])[0]
                height = unpack('>I', data[offset + 12:offset + 16])[0]
            if marker == b'pHYs':
                x_density = unpack('>I', data[offset + 8:offset + 12])[0]
                y_density = unpack('>I', data[offset + 12:offset + 16])[0]
                units = unpack('b', data[offset + 16:offset + 17])[0]
                if units == 1:
                    x_dpi = x_density * 0.0254
                    y_dpi = y_density * 0.0254
            if marker == b'IEND':
                end_marker = True
                continue
            offset = offset + length + 12
        return ('png', width, height, x_dpi, y_dpi)

    def _process_jpg(self, data):
        if False:
            i = 10
            return i + 15
        offset = 2
        data_length = len(data)
        end_marker = False
        width = 0
        height = 0
        x_dpi = 96
        y_dpi = 96
        while not end_marker and offset < data_length:
            marker = unpack('>H', data[offset + 0:offset + 2])[0]
            length = unpack('>H', data[offset + 2:offset + 4])[0]
            if marker & 65520 == 65472 and marker != 65476 and (marker != 65480) and (marker != 65484):
                height = unpack('>H', data[offset + 5:offset + 7])[0]
                width = unpack('>H', data[offset + 7:offset + 9])[0]
            if marker == 65504:
                units = unpack('b', data[offset + 11:offset + 12])[0]
                x_density = unpack('>H', data[offset + 12:offset + 14])[0]
                y_density = unpack('>H', data[offset + 14:offset + 16])[0]
                if units == 1:
                    x_dpi = x_density
                    y_dpi = y_density
                if units == 2:
                    x_dpi = x_density * 2.54
                    y_dpi = y_density * 2.54
                if x_dpi == 1:
                    x_dpi = 96
                if y_dpi == 1:
                    y_dpi = 96
            if marker == 65498:
                end_marker = True
                continue
            offset = offset + length + 2
        return ('jpeg', width, height, x_dpi, y_dpi)

    def _process_gif(self, data):
        if False:
            return 10
        x_dpi = 96
        y_dpi = 96
        width = unpack('<h', data[6:8])[0]
        height = unpack('<h', data[8:10])[0]
        return ('gif', width, height, x_dpi, y_dpi)

    def _process_bmp(self, data):
        if False:
            print('Hello World!')
        width = unpack('<L', data[18:22])[0]
        height = unpack('<L', data[22:26])[0]
        return ('bmp', width, height)

    def _process_wmf(self, data):
        if False:
            return 10
        x_dpi = 96
        y_dpi = 96
        x1 = unpack('<h', data[6:8])[0]
        y1 = unpack('<h', data[8:10])[0]
        x2 = unpack('<h', data[10:12])[0]
        y2 = unpack('<h', data[12:14])[0]
        inch = unpack('<H', data[14:16])[0]
        width = float((x2 - x1) * x_dpi) / inch
        height = float((y2 - y1) * y_dpi) / inch
        return ('wmf', width, height, x_dpi, y_dpi)

    def _process_emf(self, data):
        if False:
            return 10
        bound_x1 = unpack('<l', data[8:12])[0]
        bound_y1 = unpack('<l', data[12:16])[0]
        bound_x2 = unpack('<l', data[16:20])[0]
        bound_y2 = unpack('<l', data[20:24])[0]
        width = bound_x2 - bound_x1
        height = bound_y2 - bound_y1
        frame_x1 = unpack('<l', data[24:28])[0]
        frame_y1 = unpack('<l', data[28:32])[0]
        frame_x2 = unpack('<l', data[32:36])[0]
        frame_y2 = unpack('<l', data[36:40])[0]
        width_mm = 0.01 * (frame_x2 - frame_x1)
        height_mm = 0.01 * (frame_y2 - frame_y1)
        x_dpi = width * 25.4 / width_mm
        y_dpi = height * 25.4 / height_mm
        width += 1
        height += 1
        return ('emf', width, height, x_dpi, y_dpi)

    def _extract_named_ranges(self, defined_names):
        if False:
            print('Hello World!')
        named_ranges = []
        for defined_name in defined_names:
            name = defined_name[0]
            index = defined_name[1]
            sheet_range = defined_name[2]
            if name == '_xlnm._FilterDatabase':
                continue
            if '!' in sheet_range:
                (sheet_name, _) = sheet_range.split('!', 1)
                if name.startswith('_xlnm.'):
                    xlnm_type = name.replace('_xlnm.', '')
                    name = sheet_name + '!' + xlnm_type
                elif index != -1:
                    name = sheet_name + '!' + name
                named_ranges.append(name)
        return named_ranges

    def _get_sheet_index(self, sheetname):
        if False:
            for i in range(10):
                print('nop')
        sheetname = sheetname.strip("'")
        if sheetname in self.sheetnames:
            return self.sheetnames[sheetname].index
        else:
            return None

    def _prepare_vml(self):
        if False:
            print('Hello World!')
        comment_id = 0
        vml_drawing_id = 0
        vml_data_id = 1
        vml_header_id = 0
        vml_shape_id = 1024
        vml_files = 0
        comment_files = 0
        for sheet in self.worksheets():
            if not sheet.has_vml and (not sheet.has_header_vml):
                continue
            vml_files += 1
            if sheet.has_vml:
                if sheet.has_comments:
                    comment_files += 1
                    comment_id += 1
                    self.has_comments = True
                vml_drawing_id += 1
                count = sheet._prepare_vml_objects(vml_data_id, vml_shape_id, vml_drawing_id, comment_id)
                vml_data_id += 1 * int((1024 + count) / 1024)
                vml_shape_id += 1024 * int((1024 + count) / 1024)
            if sheet.has_header_vml:
                vml_header_id += 1
                vml_drawing_id += 1
                sheet._prepare_header_vml_objects(vml_header_id, vml_drawing_id)
            self.num_vml_files = vml_files
            self.num_comment_files = comment_files

    def _prepare_tables(self):
        if False:
            return 10
        table_id = 0
        seen = {}
        for sheet in self.worksheets():
            table_count = len(sheet.tables)
            if not table_count:
                continue
            sheet._prepare_tables(table_id + 1, seen)
            table_id += table_count

    def _prepare_metadata(self):
        if False:
            return 10
        for sheet in self.worksheets():
            if sheet.has_dynamic_arrays:
                self.has_metadata = True

    def _add_chart_data(self):
        if False:
            for i in range(10):
                print('nop')
        worksheets = {}
        seen_ranges = {}
        charts = []
        for worksheet in self.worksheets():
            worksheets[worksheet.name] = worksheet
        for chart in self.charts:
            charts.append(chart)
            if chart.combined:
                charts.append(chart.combined)
        for chart in charts:
            for c_range in chart.formula_ids.keys():
                r_id = chart.formula_ids[c_range]
                if chart.formula_data[r_id] is not None:
                    if c_range not in seen_ranges or seen_ranges[c_range] is None:
                        data = chart.formula_data[r_id]
                        seen_ranges[c_range] = data
                    continue
                if c_range in seen_ranges:
                    chart.formula_data[r_id] = seen_ranges[c_range]
                    continue
                (sheetname, cells) = self._get_chart_range(c_range)
                if sheetname is None:
                    continue
                if sheetname.startswith('('):
                    chart.formula_data[r_id] = []
                    seen_ranges[c_range] = []
                    continue
                if sheetname not in worksheets:
                    warn("Unknown worksheet reference '%s' in range '%s' passed to add_series()" % (sheetname, c_range))
                    chart.formula_data[r_id] = []
                    seen_ranges[c_range] = []
                    continue
                worksheet = worksheets[sheetname]
                data = worksheet._get_range_data(*cells)
                chart.formula_data[r_id] = data
                seen_ranges[c_range] = data

    def _get_chart_range(self, c_range):
        if False:
            print('Hello World!')
        pos = c_range.rfind('!')
        if pos > 0:
            sheetname = c_range[:pos]
            cells = c_range[pos + 1:]
        else:
            return (None, None)
        if cells.find(':') > 0:
            (cell_1, cell_2) = cells.split(':', 1)
        else:
            (cell_1, cell_2) = (cells, cells)
        sheetname = sheetname.strip("'")
        sheetname = sheetname.replace("''", "'")
        try:
            (row_start, col_start) = xl_cell_to_rowcol(cell_1)
            (row_end, col_end) = xl_cell_to_rowcol(cell_2)
        except AttributeError:
            return (None, None)
        if row_start != row_end and col_start != col_end:
            return (None, None)
        return (sheetname, [row_start, col_start, row_end, col_end])

    def _prepare_sst_string_data(self):
        if False:
            return 10
        self.str_table._sort_string_data()

    def _get_packager(self):
        if False:
            print('Hello World!')
        return Packager()

    def _write_workbook(self):
        if False:
            return 10
        schema = 'http://schemas.openxmlformats.org'
        xmlns = schema + '/spreadsheetml/2006/main'
        xmlns_r = schema + '/officeDocument/2006/relationships'
        attributes = [('xmlns', xmlns), ('xmlns:r', xmlns_r)]
        self._xml_start_tag('workbook', attributes)

    def _write_file_version(self):
        if False:
            print('Hello World!')
        app_name = 'xl'
        last_edited = 4
        lowest_edited = 4
        rup_build = 4505
        attributes = [('appName', app_name), ('lastEdited', last_edited), ('lowestEdited', lowest_edited), ('rupBuild', rup_build)]
        if self.vba_project:
            attributes.append(('codeName', '{37E998C4-C9E5-D4B9-71C8-EB1FF731991C}'))
        self._xml_empty_tag('fileVersion', attributes)

    def _write_file_sharing(self):
        if False:
            i = 10
            return i + 15
        if self.read_only == 0:
            return
        attributes = [('readOnlyRecommended', 1)]
        self._xml_empty_tag('fileSharing', attributes)

    def _write_workbook_pr(self):
        if False:
            i = 10
            return i + 15
        default_theme_version = 124226
        attributes = []
        if self.vba_codename:
            attributes.append(('codeName', self.vba_codename))
        if self.date_1904:
            attributes.append(('date1904', 1))
        attributes.append(('defaultThemeVersion', default_theme_version))
        self._xml_empty_tag('workbookPr', attributes)

    def _write_book_views(self):
        if False:
            print('Hello World!')
        self._xml_start_tag('bookViews')
        self._write_workbook_view()
        self._xml_end_tag('bookViews')

    def _write_workbook_view(self):
        if False:
            i = 10
            return i + 15
        attributes = [('xWindow', self.x_window), ('yWindow', self.y_window), ('windowWidth', self.window_width), ('windowHeight', self.window_height)]
        if self.tab_ratio != 600:
            attributes.append(('tabRatio', self.tab_ratio))
        if self.worksheet_meta.firstsheet > 0:
            firstsheet = self.worksheet_meta.firstsheet + 1
            attributes.append(('firstSheet', firstsheet))
        if self.worksheet_meta.activesheet > 0:
            attributes.append(('activeTab', self.worksheet_meta.activesheet))
        self._xml_empty_tag('workbookView', attributes)

    def _write_sheets(self):
        if False:
            for i in range(10):
                print('nop')
        self._xml_start_tag('sheets')
        id_num = 1
        for worksheet in self.worksheets():
            self._write_sheet(worksheet.name, id_num, worksheet.hidden)
            id_num += 1
        self._xml_end_tag('sheets')

    def _write_sheet(self, name, sheet_id, hidden):
        if False:
            i = 10
            return i + 15
        attributes = [('name', name), ('sheetId', sheet_id)]
        if hidden == 1:
            attributes.append(('state', 'hidden'))
        elif hidden == 2:
            attributes.append(('state', 'veryHidden'))
        attributes.append(('r:id', 'rId' + str(sheet_id)))
        self._xml_empty_tag('sheet', attributes)

    def _write_calc_pr(self):
        if False:
            i = 10
            return i + 15
        attributes = [('calcId', self.calc_id)]
        if self.calc_mode == 'manual':
            attributes.append(('calcMode', self.calc_mode))
            attributes.append(('calcOnSave', '0'))
        elif self.calc_mode == 'autoNoTable':
            attributes.append(('calcMode', self.calc_mode))
        if self.calc_on_load:
            attributes.append(('fullCalcOnLoad', '1'))
        self._xml_empty_tag('calcPr', attributes)

    def _write_defined_names(self):
        if False:
            print('Hello World!')
        if not self.defined_names:
            return
        self._xml_start_tag('definedNames')
        for defined_name in self.defined_names:
            self._write_defined_name(defined_name)
        self._xml_end_tag('definedNames')

    def _write_defined_name(self, defined_name):
        if False:
            i = 10
            return i + 15
        name = defined_name[0]
        sheet_id = defined_name[1]
        sheet_range = defined_name[2]
        hidden = defined_name[3]
        attributes = [('name', name)]
        if sheet_id != -1:
            attributes.append(('localSheetId', sheet_id))
        if hidden:
            attributes.append(('hidden', 1))
        self._xml_data_element('definedName', sheet_range, attributes)

class WorksheetMeta(object):
    """
    A class to track worksheets data such as the active sheet and the
    first sheet.

    """

    def __init__(self):
        if False:
            return 10
        self.activesheet = 0
        self.firstsheet = 0