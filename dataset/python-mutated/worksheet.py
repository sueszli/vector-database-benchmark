import datetime
import math
import os
import re
import tempfile
from collections import defaultdict
from collections import namedtuple
from decimal import Decimal
from fractions import Fraction
from functools import wraps
from io import StringIO
from math import isinf
from math import isnan
from warnings import warn
from . import xmlwriter
from .format import Format
from .drawing import Drawing
from .shape import Shape
from .xmlwriter import XMLwriter
from .utility import xl_rowcol_to_cell
from .utility import xl_rowcol_to_cell_fast
from .utility import xl_cell_to_rowcol
from .utility import xl_col_to_name
from .utility import xl_range
from .utility import xl_color
from .utility import xl_pixel_width
from .utility import get_sparkline_style
from .utility import supported_datetime
from .utility import datetime_to_excel_datetime
from .utility import preserve_whitespace
from .utility import quote_sheetname
from .exceptions import DuplicateTableName
from .exceptions import OverlappingRange
re_dynamic_function = re.compile('\n    \\bANCHORARRAY\\(    |\n    \\bBYCOL\\(          |\n    \\bBYROW\\(          |\n    \\bCHOOSECOLS\\(     |\n    \\bCHOOSEROWS\\(     |\n    \\bDROP\\(           |\n    \\bEXPAND\\(         |\n    \\bFILTER\\(         |\n    \\bHSTACK\\(         |\n    \\bLAMBDA\\(         |\n    \\bMAKEARRAY\\(      |\n    \\bMAP\\(            |\n    \\bRANDARRAY\\(      |\n    \\bREDUCE\\(         |\n    \\bSCAN\\(           |\n    \\bSEQUENCE\\(       |\n    \\bSINGLE\\(         |\n    \\bSORT\\(           |\n    \\bSORTBY\\(         |\n    \\bSWITCH\\(         |\n    \\bTAKE\\(           |\n    \\bTEXTSPLIT\\(      |\n    \\bTOCOL\\(          |\n    \\bTOROW\\(          |\n    \\bUNIQUE\\(         |\n    \\bVSTACK\\(         |\n    \\bWRAPCOLS\\(       |\n    \\bWRAPROWS\\(       |\n    \\bXLOOKUP\\(', re.VERBOSE)

def convert_cell_args(method):
    if False:
        for i in range(10):
            print('nop')
    '\n    Decorator function to convert A1 notation in cell method calls\n    to the default row/col notation.\n\n    '

    @wraps(method)
    def cell_wrapper(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        try:
            if args:
                first_arg = args[0]
                int(first_arg)
        except ValueError:
            new_args = xl_cell_to_rowcol(first_arg)
            args = new_args + args[1:]
        return method(self, *args, **kwargs)
    return cell_wrapper

def convert_range_args(method):
    if False:
        i = 10
        return i + 15
    '\n    Decorator function to convert A1 notation in range method calls\n    to the default row/col notation.\n\n    '

    @wraps(method)
    def cell_wrapper(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        try:
            if args:
                int(args[0])
        except ValueError:
            if ':' in args[0]:
                (cell_1, cell_2) = args[0].split(':')
                (row_1, col_1) = xl_cell_to_rowcol(cell_1)
                (row_2, col_2) = xl_cell_to_rowcol(cell_2)
            else:
                (row_1, col_1) = xl_cell_to_rowcol(args[0])
                (row_2, col_2) = (row_1, col_1)
            new_args = [row_1, col_1, row_2, col_2]
            new_args.extend(args[1:])
            args = new_args
        return method(self, *args, **kwargs)
    return cell_wrapper

def convert_column_args(method):
    if False:
        print('Hello World!')
    '\n    Decorator function to convert A1 notation in columns method calls\n    to the default row/col notation.\n\n    '

    @wraps(method)
    def column_wrapper(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        try:
            if args:
                int(args[0])
        except ValueError:
            (cell_1, cell_2) = [col + '1' for col in args[0].split(':')]
            (_, col_1) = xl_cell_to_rowcol(cell_1)
            (_, col_2) = xl_cell_to_rowcol(cell_2)
            new_args = [col_1, col_2]
            new_args.extend(args[1:])
            args = new_args
        return method(self, *args, **kwargs)
    return column_wrapper
cell_string_tuple = namedtuple('String', 'string, format')
cell_number_tuple = namedtuple('Number', 'number, format')
cell_blank_tuple = namedtuple('Blank', 'format')
cell_boolean_tuple = namedtuple('Boolean', 'boolean, format')
cell_formula_tuple = namedtuple('Formula', 'formula, format, value')
cell_datetime_tuple = namedtuple('Datetime', 'number, format')
cell_arformula_tuple = namedtuple('ArrayFormula', 'formula, format, value, range, atype')
cell_rich_string_tuple = namedtuple('RichString', 'string, format, raw_string')

class Worksheet(xmlwriter.XMLwriter):
    """
    A class for writing the Excel XLSX Worksheet file.

    """

    def __init__(self):
        if False:
            return 10
        '\n        Constructor.\n\n        '
        super(Worksheet, self).__init__()
        self.name = None
        self.index = None
        self.str_table = None
        self.palette = None
        self.constant_memory = 0
        self.tmpdir = None
        self.is_chartsheet = False
        self.ext_sheets = []
        self.fileclosed = 0
        self.excel_version = 2007
        self.excel2003_style = False
        self.xls_rowmax = 1048576
        self.xls_colmax = 16384
        self.xls_strmax = 32767
        self.dim_rowmin = None
        self.dim_rowmax = None
        self.dim_colmin = None
        self.dim_colmax = None
        self.col_info = {}
        self.selections = []
        self.hidden = 0
        self.active = 0
        self.tab_color = 0
        self.top_left_cell = ''
        self.panes = []
        self.active_pane = 3
        self.selected = 0
        self.page_setup_changed = False
        self.paper_size = 0
        self.orientation = 1
        self.print_options_changed = False
        self.hcenter = False
        self.vcenter = False
        self.print_gridlines = False
        self.screen_gridlines = True
        self.print_headers = False
        self.row_col_headers = False
        self.header_footer_changed = False
        self.header = ''
        self.footer = ''
        self.header_footer_aligns = True
        self.header_footer_scales = True
        self.header_images = []
        self.footer_images = []
        self.header_images_list = []
        self.margin_left = 0.7
        self.margin_right = 0.7
        self.margin_top = 0.75
        self.margin_bottom = 0.75
        self.margin_header = 0.3
        self.margin_footer = 0.3
        self.repeat_row_range = ''
        self.repeat_col_range = ''
        self.print_area_range = ''
        self.page_order = 0
        self.black_white = 0
        self.draft_quality = 0
        self.print_comments = 0
        self.page_start = 0
        self.fit_page = 0
        self.fit_width = 0
        self.fit_height = 0
        self.hbreaks = []
        self.vbreaks = []
        self.protect_options = {}
        self.protected_ranges = []
        self.num_protected_ranges = 0
        self.set_cols = {}
        self.set_rows = defaultdict(dict)
        self.zoom = 100
        self.zoom_scale_normal = 1
        self.print_scale = 100
        self.is_right_to_left = 0
        self.show_zeros = 1
        self.leading_zeros = 0
        self.outline_row_level = 0
        self.outline_col_level = 0
        self.outline_style = 0
        self.outline_below = 1
        self.outline_right = 1
        self.outline_on = 1
        self.outline_changed = False
        self.original_row_height = 15
        self.default_row_height = 15
        self.default_row_pixels = 20
        self.default_col_width = 8.43
        self.default_col_pixels = 64
        self.default_date_pixels = 68
        self.default_row_zeroed = 0
        self.names = {}
        self.write_match = []
        self.table = defaultdict(dict)
        self.merge = []
        self.merged_cells = {}
        self.table_cells = {}
        self.row_spans = {}
        self.has_vml = False
        self.has_header_vml = False
        self.has_comments = False
        self.comments = defaultdict(dict)
        self.comments_list = []
        self.comments_author = ''
        self.comments_visible = 0
        self.vml_shape_id = 1024
        self.buttons_list = []
        self.vml_header_id = 0
        self.autofilter_area = ''
        self.autofilter_ref = None
        self.filter_range = []
        self.filter_on = 0
        self.filter_cols = {}
        self.filter_type = {}
        self.filter_cells = {}
        self.row_sizes = {}
        self.col_size_changed = False
        self.row_size_changed = False
        self.last_shape_id = 1
        self.rel_count = 0
        self.hlink_count = 0
        self.hlink_refs = []
        self.external_hyper_links = []
        self.external_drawing_links = []
        self.external_comment_links = []
        self.external_vml_links = []
        self.external_table_links = []
        self.external_background_links = []
        self.drawing_links = []
        self.vml_drawing_links = []
        self.charts = []
        self.images = []
        self.tables = []
        self.sparklines = []
        self.shapes = []
        self.shape_hash = {}
        self.drawing = 0
        self.drawing_rels = {}
        self.drawing_rels_id = 0
        self.vml_drawing_rels = {}
        self.vml_drawing_rels_id = 0
        self.background_image = None
        self.background_bytes = False
        self.rstring = ''
        self.previous_row = 0
        self.validations = []
        self.cond_formats = {}
        self.data_bars_2010 = []
        self.use_data_bars_2010 = False
        self.dxf_priority = 1
        self.page_view = 0
        self.vba_codename = None
        self.date_1904 = False
        self.hyperlinks = defaultdict(dict)
        self.strings_to_numbers = False
        self.strings_to_urls = True
        self.nan_inf_to_errors = False
        self.strings_to_formulas = True
        self.default_date_format = None
        self.default_url_format = None
        self.remove_timezone = False
        self.max_url_length = 2079
        self.row_data_filename = None
        self.row_data_fh = None
        self.worksheet_meta = None
        self.vml_data_id = None
        self.vml_shape_id = None
        self.row_data_filename = None
        self.row_data_fh = None
        self.row_data_fh_closed = False
        self.vertical_dpi = 0
        self.horizontal_dpi = 0
        self.write_handlers = {}
        self.ignored_errors = None
        self.has_dynamic_arrays = False
        self.use_future_functions = False

    def _write_token_as_string(self, token, row, col, *args):
        if False:
            i = 10
            return i + 15
        if token == '':
            return self._write_blank(row, col, *args)
        if self.strings_to_formulas and token.startswith('='):
            return self._write_formula(row, col, *args)
        if token.startswith('{=') and token.endswith('}'):
            return self._write_formula(row, col, *args)
        if ':' in token and self.strings_to_urls and (re.match('(ftp|http)s?://', token) or re.match('mailto:', token) or re.match('(in|ex)ternal:', token)):
            return self._write_url(row, col, *args)
        if self.strings_to_numbers:
            try:
                f = float(token)
                if self.nan_inf_to_errors or (not isnan(f) and (not isinf(f))):
                    return self._write_number(row, col, f, *args[1:])
            except ValueError:
                pass
            return self._write_string(row, col, *args)
        else:
            return self._write_string(row, col, *args)

    @convert_cell_args
    def write(self, row, col, *args):
        if False:
            print('Hello World!')
        '\n        Write data to a worksheet cell by calling the appropriate write_*()\n        method based on the type of data being passed.\n\n        Args:\n            row:   The cell row (zero indexed).\n            col:   The cell column (zero indexed).\n            *args: Args to pass to sub functions.\n\n        Returns:\n             0:    Success.\n            -1:    Row or column is out of worksheet bounds.\n            other: Return value of called method.\n\n        '
        return self._write(row, col, *args)

    def _write(self, row, col, *args):
        if False:
            while True:
                i = 10
        if not args:
            raise TypeError('write() takes at least 4 arguments (3 given)')
        token = args[0]
        token_type = token.__class__
        if token_type in self.write_handlers:
            write_handler = self.write_handlers[token_type]
            function_return = write_handler(self, row, col, *args)
            if function_return is None:
                pass
            else:
                return function_return
        if token is None:
            return self._write_blank(row, col, *args)
        if token_type is bool:
            return self._write_boolean(row, col, *args)
        if token_type in (float, int, Decimal, Fraction):
            return self._write_number(row, col, *args)
        if token_type is str:
            return self._write_token_as_string(token, row, col, *args)
        if token_type in (datetime.datetime, datetime.date, datetime.time, datetime.timedelta):
            return self._write_datetime(row, col, *args)
        if isinstance(token, (float, int, Decimal, Fraction)):
            return self._write_number(row, col, *args)
        if isinstance(token, str):
            return self._write_token_as_string(token, row, col, *args)
        if isinstance(token, bool):
            return self._write_boolean(row, col, *args)
        if supported_datetime(token):
            return self._write_datetime(row, col, *args)
        try:
            f = float(token)
            return self._write_number(row, col, f, *args[1:])
        except ValueError:
            pass
        except TypeError:
            raise TypeError('Unsupported type %s in write()' % type(token))
        try:
            str(token)
            return self._write_string(row, col, *args)
        except ValueError:
            raise TypeError('Unsupported type %s in write()' % type(token))

    @convert_cell_args
    def write_string(self, row, col, string, cell_format=None):
        if False:
            print('Hello World!')
        '\n        Write a string to a worksheet cell.\n\n        Args:\n            row:    The cell row (zero indexed).\n            col:    The cell column (zero indexed).\n            string: Cell data. Str.\n            format: An optional cell Format object.\n\n        Returns:\n            0:  Success.\n            -1: Row or column is out of worksheet bounds.\n            -2: String truncated to 32k characters.\n\n        '
        return self._write_string(row, col, string, cell_format)

    def _write_string(self, row, col, string, cell_format=None):
        if False:
            return 10
        str_error = 0
        if self._check_dimensions(row, col):
            return -1
        if len(string) > self.xls_strmax:
            string = string[:self.xls_strmax]
            str_error = -2
        if not self.constant_memory:
            string_index = self.str_table._get_shared_string_index(string)
        else:
            string_index = string
        if self.constant_memory and row > self.previous_row:
            self._write_single_row(row)
        self.table[row][col] = cell_string_tuple(string_index, cell_format)
        return str_error

    @convert_cell_args
    def write_number(self, row, col, number, cell_format=None):
        if False:
            return 10
        '\n        Write a number to a worksheet cell.\n\n        Args:\n            row:         The cell row (zero indexed).\n            col:         The cell column (zero indexed).\n            number:      Cell data. Int or float.\n            cell_format: An optional cell Format object.\n\n        Returns:\n            0:  Success.\n            -1: Row or column is out of worksheet bounds.\n\n        '
        return self._write_number(row, col, number, cell_format)

    def _write_number(self, row, col, number, cell_format=None):
        if False:
            return 10
        if isnan(number) or isinf(number):
            if self.nan_inf_to_errors:
                if isnan(number):
                    return self._write_formula(row, col, '#NUM!', cell_format, '#NUM!')
                elif number == math.inf:
                    return self._write_formula(row, col, '1/0', cell_format, '#DIV/0!')
                elif number == -math.inf:
                    return self._write_formula(row, col, '-1/0', cell_format, '#DIV/0!')
            else:
                raise TypeError("NAN/INF not supported in write_number() without 'nan_inf_to_errors' Workbook() option")
        if self._check_dimensions(row, col):
            return -1
        if self.constant_memory and row > self.previous_row:
            self._write_single_row(row)
        self.table[row][col] = cell_number_tuple(number, cell_format)
        return 0

    @convert_cell_args
    def write_blank(self, row, col, blank, cell_format=None):
        if False:
            print('Hello World!')
        '\n        Write a blank cell with formatting to a worksheet cell. The blank\n        token is ignored and the format only is written to the cell.\n\n        Args:\n            row:         The cell row (zero indexed).\n            col:         The cell column (zero indexed).\n            blank:       Any value. It is ignored.\n            cell_format: An optional cell Format object.\n\n        Returns:\n            0:  Success.\n            -1: Row or column is out of worksheet bounds.\n\n        '
        return self._write_blank(row, col, blank, cell_format)

    def _write_blank(self, row, col, blank, cell_format=None):
        if False:
            while True:
                i = 10
        if cell_format is None:
            return 0
        if self._check_dimensions(row, col):
            return -1
        if self.constant_memory and row > self.previous_row:
            self._write_single_row(row)
        self.table[row][col] = cell_blank_tuple(cell_format)
        return 0

    @convert_cell_args
    def write_formula(self, row, col, formula, cell_format=None, value=0):
        if False:
            i = 10
            return i + 15
        "\n        Write a formula to a worksheet cell.\n\n        Args:\n            row:         The cell row (zero indexed).\n            col:         The cell column (zero indexed).\n            formula:     Cell formula.\n            cell_format: An optional cell Format object.\n            value:       An optional value for the formula. Default is 0.\n\n        Returns:\n            0:  Success.\n            -1: Row or column is out of worksheet bounds.\n            -2: Formula can't be None or empty.\n\n        "
        return self._write_formula(row, col, formula, cell_format, value)

    def _write_formula(self, row, col, formula, cell_format=None, value=0):
        if False:
            while True:
                i = 10
        if self._check_dimensions(row, col):
            return -1
        if formula is None or formula == '':
            warn("Formula can't be None or empty")
            return -1
        if re_dynamic_function.search(formula):
            return self.write_dynamic_array_formula(row, col, row, col, formula, cell_format, value)
        if formula.startswith('{') and formula.endswith('}'):
            return self._write_array_formula(row, col, row, col, formula, cell_format, value)
        formula = self._prepare_formula(formula)
        if self.constant_memory and row > self.previous_row:
            self._write_single_row(row)
        self.table[row][col] = cell_formula_tuple(formula, cell_format, value)
        return 0

    @convert_range_args
    def write_array_formula(self, first_row, first_col, last_row, last_col, formula, cell_format=None, value=0):
        if False:
            print('Hello World!')
        '\n        Write a formula to a worksheet cell/range.\n\n        Args:\n            first_row:    The first row of the cell range. (zero indexed).\n            first_col:    The first column of the cell range.\n            last_row:     The last row of the cell range. (zero indexed).\n            last_col:     The last column of the cell range.\n            formula:      Cell formula.\n            cell_format:  An optional cell Format object.\n            value:        An optional value for the formula. Default is 0.\n\n        Returns:\n            0:  Success.\n            -1: Row or column is out of worksheet bounds.\n\n        '
        if re_dynamic_function.search(formula):
            return self.write_dynamic_array_formula(first_row, first_col, last_row, last_col, formula, cell_format, value)
        return self._write_array_formula(first_row, first_col, last_row, last_col, formula, cell_format, value, 'static')

    @convert_range_args
    def write_dynamic_array_formula(self, first_row, first_col, last_row, last_col, formula, cell_format=None, value=0):
        if False:
            while True:
                i = 10
        '\n        Write a dynamic array formula to a worksheet cell/range.\n\n        Args:\n            first_row:    The first row of the cell range. (zero indexed).\n            first_col:    The first column of the cell range.\n            last_row:     The last row of the cell range. (zero indexed).\n            last_col:     The last column of the cell range.\n            formula:      Cell formula.\n            cell_format:  An optional cell Format object.\n            value:        An optional value for the formula. Default is 0.\n\n        Returns:\n            0:  Success.\n            -1: Row or column is out of worksheet bounds.\n\n        '
        error = self._write_array_formula(first_row, first_col, last_row, last_col, formula, cell_format, value, 'dynamic')
        if error == 0:
            self.has_dynamic_arrays = True
        return error

    def _prepare_formula(self, formula, expand_future_functions=False):
        if False:
            return 10
        if formula.startswith('{'):
            formula = formula[1:]
        if formula.startswith('='):
            formula = formula[1:]
        if formula.endswith('}'):
            formula = formula[:-1]
        if '_xlfn.' in formula:
            return formula
        formula = re.sub('\\bANCHORARRAY\\(', '_xlfn.ANCHORARRAY(', formula)
        formula = re.sub('\\bBYCOL\\(', '_xlfn.BYCOL(', formula)
        formula = re.sub('\\bBYROW\\(', '_xlfn.BYROW(', formula)
        formula = re.sub('\\bCHOOSECOLS\\(', '_xlfn.CHOOSECOLS(', formula)
        formula = re.sub('\\bCHOOSEROWS\\(', '_xlfn.CHOOSEROWS(', formula)
        formula = re.sub('\\bDROP\\(', '_xlfn.DROP(', formula)
        formula = re.sub('\\bEXPAND\\(', '_xlfn.EXPAND(', formula)
        formula = re.sub('\\bFILTER\\(', '_xlfn._xlws.FILTER(', formula)
        formula = re.sub('\\bHSTACK\\(', '_xlfn.HSTACK(', formula)
        formula = re.sub('\\bLAMBDA\\(', '_xlfn.LAMBDA(', formula)
        formula = re.sub('\\bMAKEARRAY\\(', '_xlfn.MAKEARRAY(', formula)
        formula = re.sub('\\bMAP\\(', '_xlfn.MAP(', formula)
        formula = re.sub('\\bRANDARRAY\\(', '_xlfn.RANDARRAY(', formula)
        formula = re.sub('\\bREDUCE\\(', '_xlfn.REDUCE(', formula)
        formula = re.sub('\\bSCAN\\(', '_xlfn.SCAN(', formula)
        formula = re.sub('\\SINGLE\\(', '_xlfn.SINGLE(', formula)
        formula = re.sub('\\bSEQUENCE\\(', '_xlfn.SEQUENCE(', formula)
        formula = re.sub('\\bSORT\\(', '_xlfn._xlws.SORT(', formula)
        formula = re.sub('\\bSORTBY\\(', '_xlfn.SORTBY(', formula)
        formula = re.sub('\\bSWITCH\\(', '_xlfn.SWITCH(', formula)
        formula = re.sub('\\bTAKE\\(', '_xlfn.TAKE(', formula)
        formula = re.sub('\\bTEXTSPLIT\\(', '_xlfn.TEXTSPLIT(', formula)
        formula = re.sub('\\bTOCOL\\(', '_xlfn.TOCOL(', formula)
        formula = re.sub('\\bTOROW\\(', '_xlfn.TOROW(', formula)
        formula = re.sub('\\bUNIQUE\\(', '_xlfn.UNIQUE(', formula)
        formula = re.sub('\\bVSTACK\\(', '_xlfn.VSTACK(', formula)
        formula = re.sub('\\bWRAPCOLS\\(', '_xlfn.WRAPCOLS(', formula)
        formula = re.sub('\\bWRAPROWS\\(', '_xlfn.WRAPROWS(', formula)
        formula = re.sub('\\bXLOOKUP\\(', '_xlfn.XLOOKUP(', formula)
        if not self.use_future_functions and (not expand_future_functions):
            return formula
        formula = re.sub('\\bACOTH\\(', '_xlfn.ACOTH(', formula)
        formula = re.sub('\\bACOT\\(', '_xlfn.ACOT(', formula)
        formula = re.sub('\\bAGGREGATE\\(', '_xlfn.AGGREGATE(', formula)
        formula = re.sub('\\bARABIC\\(', '_xlfn.ARABIC(', formula)
        formula = re.sub('\\bARRAYTOTEXT\\(', '_xlfn.ARRAYTOTEXT(', formula)
        formula = re.sub('\\bBASE\\(', '_xlfn.BASE(', formula)
        formula = re.sub('\\bBETA.DIST\\(', '_xlfn.BETA.DIST(', formula)
        formula = re.sub('\\bBETA.INV\\(', '_xlfn.BETA.INV(', formula)
        formula = re.sub('\\bBINOM.DIST.RANGE\\(', '_xlfn.BINOM.DIST.RANGE(', formula)
        formula = re.sub('\\bBINOM.DIST\\(', '_xlfn.BINOM.DIST(', formula)
        formula = re.sub('\\bBINOM.INV\\(', '_xlfn.BINOM.INV(', formula)
        formula = re.sub('\\bBITAND\\(', '_xlfn.BITAND(', formula)
        formula = re.sub('\\bBITLSHIFT\\(', '_xlfn.BITLSHIFT(', formula)
        formula = re.sub('\\bBITOR\\(', '_xlfn.BITOR(', formula)
        formula = re.sub('\\bBITRSHIFT\\(', '_xlfn.BITRSHIFT(', formula)
        formula = re.sub('\\bBITXOR\\(', '_xlfn.BITXOR(', formula)
        formula = re.sub('\\bCEILING.MATH\\(', '_xlfn.CEILING.MATH(', formula)
        formula = re.sub('\\bCEILING.PRECISE\\(', '_xlfn.CEILING.PRECISE(', formula)
        formula = re.sub('\\bCHISQ.DIST.RT\\(', '_xlfn.CHISQ.DIST.RT(', formula)
        formula = re.sub('\\bCHISQ.DIST\\(', '_xlfn.CHISQ.DIST(', formula)
        formula = re.sub('\\bCHISQ.INV.RT\\(', '_xlfn.CHISQ.INV.RT(', formula)
        formula = re.sub('\\bCHISQ.INV\\(', '_xlfn.CHISQ.INV(', formula)
        formula = re.sub('\\bCHISQ.TEST\\(', '_xlfn.CHISQ.TEST(', formula)
        formula = re.sub('\\bCOMBINA\\(', '_xlfn.COMBINA(', formula)
        formula = re.sub('\\bCONCAT\\(', '_xlfn.CONCAT(', formula)
        formula = re.sub('\\bCONFIDENCE.NORM\\(', '_xlfn.CONFIDENCE.NORM(', formula)
        formula = re.sub('\\bCONFIDENCE.T\\(', '_xlfn.CONFIDENCE.T(', formula)
        formula = re.sub('\\bCOTH\\(', '_xlfn.COTH(', formula)
        formula = re.sub('\\bCOT\\(', '_xlfn.COT(', formula)
        formula = re.sub('\\bCOVARIANCE.P\\(', '_xlfn.COVARIANCE.P(', formula)
        formula = re.sub('\\bCOVARIANCE.S\\(', '_xlfn.COVARIANCE.S(', formula)
        formula = re.sub('\\bCSCH\\(', '_xlfn.CSCH(', formula)
        formula = re.sub('\\bCSC\\(', '_xlfn.CSC(', formula)
        formula = re.sub('\\bDAYS\\(', '_xlfn.DAYS(', formula)
        formula = re.sub('\\bDECIMAL\\(', '_xlfn.DECIMAL(', formula)
        formula = re.sub('\\bERF.PRECISE\\(', '_xlfn.ERF.PRECISE(', formula)
        formula = re.sub('\\bERFC.PRECISE\\(', '_xlfn.ERFC.PRECISE(', formula)
        formula = re.sub('\\bEXPON.DIST\\(', '_xlfn.EXPON.DIST(', formula)
        formula = re.sub('\\bF.DIST.RT\\(', '_xlfn.F.DIST.RT(', formula)
        formula = re.sub('\\bF.DIST\\(', '_xlfn.F.DIST(', formula)
        formula = re.sub('\\bF.INV.RT\\(', '_xlfn.F.INV.RT(', formula)
        formula = re.sub('\\bF.INV\\(', '_xlfn.F.INV(', formula)
        formula = re.sub('\\bF.TEST\\(', '_xlfn.F.TEST(', formula)
        formula = re.sub('\\bFILTERXML\\(', '_xlfn.FILTERXML(', formula)
        formula = re.sub('\\bFLOOR.MATH\\(', '_xlfn.FLOOR.MATH(', formula)
        formula = re.sub('\\bFLOOR.PRECISE\\(', '_xlfn.FLOOR.PRECISE(', formula)
        formula = re.sub('\\bFORECAST.ETS.CONFINT\\(', '_xlfn.FORECAST.ETS.CONFINT(', formula)
        formula = re.sub('\\bFORECAST.ETS.SEASONALITY\\(', '_xlfn.FORECAST.ETS.SEASONALITY(', formula)
        formula = re.sub('\\bFORECAST.ETS.STAT\\(', '_xlfn.FORECAST.ETS.STAT(', formula)
        formula = re.sub('\\bFORECAST.ETS\\(', '_xlfn.FORECAST.ETS(', formula)
        formula = re.sub('\\bFORECAST.LINEAR\\(', '_xlfn.FORECAST.LINEAR(', formula)
        formula = re.sub('\\bFORMULATEXT\\(', '_xlfn.FORMULATEXT(', formula)
        formula = re.sub('\\bGAMMA.DIST\\(', '_xlfn.GAMMA.DIST(', formula)
        formula = re.sub('\\bGAMMA.INV\\(', '_xlfn.GAMMA.INV(', formula)
        formula = re.sub('\\bGAMMALN.PRECISE\\(', '_xlfn.GAMMALN.PRECISE(', formula)
        formula = re.sub('\\bGAMMA\\(', '_xlfn.GAMMA(', formula)
        formula = re.sub('\\bGAUSS\\(', '_xlfn.GAUSS(', formula)
        formula = re.sub('\\bHYPGEOM.DIST\\(', '_xlfn.HYPGEOM.DIST(', formula)
        formula = re.sub('\\bIFNA\\(', '_xlfn.IFNA(', formula)
        formula = re.sub('\\bIFS\\(', '_xlfn.IFS(', formula)
        formula = re.sub('\\bIMAGE\\(', '_xlfn.IMAGE(', formula)
        formula = re.sub('\\bIMCOSH\\(', '_xlfn.IMCOSH(', formula)
        formula = re.sub('\\bIMCOT\\(', '_xlfn.IMCOT(', formula)
        formula = re.sub('\\bIMCSCH\\(', '_xlfn.IMCSCH(', formula)
        formula = re.sub('\\bIMCSC\\(', '_xlfn.IMCSC(', formula)
        formula = re.sub('\\bIMSECH\\(', '_xlfn.IMSECH(', formula)
        formula = re.sub('\\bIMSEC\\(', '_xlfn.IMSEC(', formula)
        formula = re.sub('\\bIMSINH\\(', '_xlfn.IMSINH(', formula)
        formula = re.sub('\\bIMTAN\\(', '_xlfn.IMTAN(', formula)
        formula = re.sub('\\bISFORMULA\\(', '_xlfn.ISFORMULA(', formula)
        formula = re.sub('\\bISOMITTED\\(', '_xlfn.ISOMITTED(', formula)
        formula = re.sub('\\bISOWEEKNUM\\(', '_xlfn.ISOWEEKNUM(', formula)
        formula = re.sub('\\bLET\\(', '_xlfn.LET(', formula)
        formula = re.sub('\\bLOGNORM.DIST\\(', '_xlfn.LOGNORM.DIST(', formula)
        formula = re.sub('\\bLOGNORM.INV\\(', '_xlfn.LOGNORM.INV(', formula)
        formula = re.sub('\\bMAXIFS\\(', '_xlfn.MAXIFS(', formula)
        formula = re.sub('\\bMINIFS\\(', '_xlfn.MINIFS(', formula)
        formula = re.sub('\\bMODE.MULT\\(', '_xlfn.MODE.MULT(', formula)
        formula = re.sub('\\bMODE.SNGL\\(', '_xlfn.MODE.SNGL(', formula)
        formula = re.sub('\\bMUNIT\\(', '_xlfn.MUNIT(', formula)
        formula = re.sub('\\bNEGBINOM.DIST\\(', '_xlfn.NEGBINOM.DIST(', formula)
        formula = re.sub('\\bNORM.DIST\\(', '_xlfn.NORM.DIST(', formula)
        formula = re.sub('\\bNORM.INV\\(', '_xlfn.NORM.INV(', formula)
        formula = re.sub('\\bNORM.S.DIST\\(', '_xlfn.NORM.S.DIST(', formula)
        formula = re.sub('\\bNORM.S.INV\\(', '_xlfn.NORM.S.INV(', formula)
        formula = re.sub('\\bNUMBERVALUE\\(', '_xlfn.NUMBERVALUE(', formula)
        formula = re.sub('\\bPDURATION\\(', '_xlfn.PDURATION(', formula)
        formula = re.sub('\\bPERCENTILE.EXC\\(', '_xlfn.PERCENTILE.EXC(', formula)
        formula = re.sub('\\bPERCENTILE.INC\\(', '_xlfn.PERCENTILE.INC(', formula)
        formula = re.sub('\\bPERCENTRANK.EXC\\(', '_xlfn.PERCENTRANK.EXC(', formula)
        formula = re.sub('\\bPERCENTRANK.INC\\(', '_xlfn.PERCENTRANK.INC(', formula)
        formula = re.sub('\\bPERMUTATIONA\\(', '_xlfn.PERMUTATIONA(', formula)
        formula = re.sub('\\bPHI\\(', '_xlfn.PHI(', formula)
        formula = re.sub('\\bPOISSON.DIST\\(', '_xlfn.POISSON.DIST(', formula)
        formula = re.sub('\\bQUARTILE.EXC\\(', '_xlfn.QUARTILE.EXC(', formula)
        formula = re.sub('\\bQUARTILE.INC\\(', '_xlfn.QUARTILE.INC(', formula)
        formula = re.sub('\\bQUERYSTRING\\(', '_xlfn.QUERYSTRING(', formula)
        formula = re.sub('\\bRANK.AVG\\(', '_xlfn.RANK.AVG(', formula)
        formula = re.sub('\\bRANK.EQ\\(', '_xlfn.RANK.EQ(', formula)
        formula = re.sub('\\bRRI\\(', '_xlfn.RRI(', formula)
        formula = re.sub('\\bSECH\\(', '_xlfn.SECH(', formula)
        formula = re.sub('\\bSEC\\(', '_xlfn.SEC(', formula)
        formula = re.sub('\\bSHEETS\\(', '_xlfn.SHEETS(', formula)
        formula = re.sub('\\bSHEET\\(', '_xlfn.SHEET(', formula)
        formula = re.sub('\\bSKEW.P\\(', '_xlfn.SKEW.P(', formula)
        formula = re.sub('\\bSTDEV.P\\(', '_xlfn.STDEV.P(', formula)
        formula = re.sub('\\bSTDEV.S\\(', '_xlfn.STDEV.S(', formula)
        formula = re.sub('\\bT.DIST.2T\\(', '_xlfn.T.DIST.2T(', formula)
        formula = re.sub('\\bT.DIST.RT\\(', '_xlfn.T.DIST.RT(', formula)
        formula = re.sub('\\bT.DIST\\(', '_xlfn.T.DIST(', formula)
        formula = re.sub('\\bT.INV.2T\\(', '_xlfn.T.INV.2T(', formula)
        formula = re.sub('\\bT.INV\\(', '_xlfn.T.INV(', formula)
        formula = re.sub('\\bT.TEST\\(', '_xlfn.T.TEST(', formula)
        formula = re.sub('\\bTEXTAFTER\\(', '_xlfn.TEXTAFTER(', formula)
        formula = re.sub('\\bTEXTBEFORE\\(', '_xlfn.TEXTBEFORE(', formula)
        formula = re.sub('\\bTEXTJOIN\\(', '_xlfn.TEXTJOIN(', formula)
        formula = re.sub('\\bUNICHAR\\(', '_xlfn.UNICHAR(', formula)
        formula = re.sub('\\bUNICODE\\(', '_xlfn.UNICODE(', formula)
        formula = re.sub('\\bVALUETOTEXT\\(', '_xlfn.VALUETOTEXT(', formula)
        formula = re.sub('\\bVAR.P\\(', '_xlfn.VAR.P(', formula)
        formula = re.sub('\\bVAR.S\\(', '_xlfn.VAR.S(', formula)
        formula = re.sub('\\bWEBSERVICE\\(', '_xlfn.WEBSERVICE(', formula)
        formula = re.sub('\\bWEIBULL.DIST\\(', '_xlfn.WEIBULL.DIST(', formula)
        formula = re.sub('\\bXMATCH\\(', '_xlfn.XMATCH(', formula)
        formula = re.sub('\\bXOR\\(', '_xlfn.XOR(', formula)
        formula = re.sub('\\bZ.TEST\\(', '_xlfn.Z.TEST(', formula)
        return formula

    def _write_array_formula(self, first_row, first_col, last_row, last_col, formula, cell_format=None, value=0, atype='static'):
        if False:
            print('Hello World!')
        if first_row > last_row:
            (first_row, last_row) = (last_row, first_row)
        if first_col > last_col:
            (first_col, last_col) = (last_col, first_col)
        if self._check_dimensions(first_row, first_col):
            return -1
        if self._check_dimensions(last_row, last_col):
            return -1
        if first_row == last_row and first_col == last_col:
            cell_range = xl_rowcol_to_cell(first_row, first_col)
        else:
            cell_range = xl_rowcol_to_cell(first_row, first_col) + ':' + xl_rowcol_to_cell(last_row, last_col)
        formula = self._prepare_formula(formula)
        if self.constant_memory and first_row > self.previous_row:
            self._write_single_row(first_row)
        self.table[first_row][first_col] = cell_arformula_tuple(formula, cell_format, value, cell_range, atype)
        if not self.constant_memory:
            for row in range(first_row, last_row + 1):
                for col in range(first_col, last_col + 1):
                    if row != first_row or col != first_col:
                        self._write_number(row, col, 0, cell_format)
        return 0

    @convert_cell_args
    def write_datetime(self, row, col, date, cell_format=None):
        if False:
            return 10
        '\n        Write a date or time to a worksheet cell.\n\n        Args:\n            row:         The cell row (zero indexed).\n            col:         The cell column (zero indexed).\n            date:        Date and/or time as a datetime object.\n            cell_format: A cell Format object.\n\n        Returns:\n            0:  Success.\n            -1: Row or column is out of worksheet bounds.\n\n        '
        return self._write_datetime(row, col, date, cell_format)

    def _write_datetime(self, row, col, date, cell_format=None):
        if False:
            for i in range(10):
                print('nop')
        if self._check_dimensions(row, col):
            return -1
        if self.constant_memory and row > self.previous_row:
            self._write_single_row(row)
        number = self._convert_date_time(date)
        if cell_format is None:
            cell_format = self.default_date_format
        self.table[row][col] = cell_datetime_tuple(number, cell_format)
        return 0

    @convert_cell_args
    def write_boolean(self, row, col, boolean, cell_format=None):
        if False:
            while True:
                i = 10
        '\n        Write a boolean value to a worksheet cell.\n\n        Args:\n            row:         The cell row (zero indexed).\n            col:         The cell column (zero indexed).\n            boolean:     Cell data. bool type.\n            cell_format: An optional cell Format object.\n\n        Returns:\n            0:  Success.\n            -1: Row or column is out of worksheet bounds.\n\n        '
        return self._write_boolean(row, col, boolean, cell_format)

    def _write_boolean(self, row, col, boolean, cell_format=None):
        if False:
            i = 10
            return i + 15
        if self._check_dimensions(row, col):
            return -1
        if self.constant_memory and row > self.previous_row:
            self._write_single_row(row)
        if boolean:
            value = 1
        else:
            value = 0
        self.table[row][col] = cell_boolean_tuple(value, cell_format)
        return 0

    @convert_cell_args
    def write_url(self, row, col, url, cell_format=None, string=None, tip=None):
        if False:
            return 10
        '\n        Write a hyperlink to a worksheet cell.\n\n        Args:\n            row:    The cell row (zero indexed).\n            col:    The cell column (zero indexed).\n            url:    Hyperlink url.\n            format: An optional cell Format object.\n            string: An optional display string for the hyperlink.\n            tip:    An optional tooltip.\n        Returns:\n            0:  Success.\n            -1: Row or column is out of worksheet bounds.\n            -2: String longer than 32767 characters.\n            -3: URL longer than Excel limit of 255 characters.\n            -4: Exceeds Excel limit of 65,530 urls per worksheet.\n        '
        return self._write_url(row, col, url, cell_format, string, tip)

    def _write_url(self, row, col, url, cell_format=None, string=None, tip=None):
        if False:
            return 10
        if self._check_dimensions(row, col):
            return -1
        if string is None:
            string = url
        link_type = 1
        if url.startswith('internal:'):
            url = url.replace('internal:', '')
            string = string.replace('internal:', '')
            link_type = 2
        external = False
        if url.startswith('external:'):
            url = url.replace('external:', '')
            url = url.replace('/', '\\')
            string = string.replace('external:', '')
            string = string.replace('/', '\\')
            external = True
        string = string.replace('mailto:', '')
        str_error = 0
        if len(string) > self.xls_strmax:
            warn("Ignoring URL since it exceeds Excel's string limit of 32767 characters")
            return -2
        url_str = string
        if link_type == 1:
            if '#' in url:
                (url, url_str) = url.split('#', 1)
            else:
                url_str = None
            url = self._escape_url(url)
            if url_str is not None and (not external):
                url_str = self._escape_url(url_str)
            if re.match('\\w:', url) or re.match('\\\\', url):
                url = 'file:///' + url
            url = re.sub('^\\.\\\\', '', url)
        tmp_url_str = url_str or ''
        max_url = self.max_url_length
        if len(url) > max_url or len(tmp_url_str) > max_url:
            warn("Ignoring URL '%s' with link or location/anchor > %d characters since it exceeds Excel's limit for URLS" % (url, max_url))
            return -3
        self.hlink_count += 1
        if self.hlink_count > 65530:
            warn("Ignoring URL '%s' since it exceeds Excel's limit of 65,530 URLS per worksheet." % url)
            return -4
        if self.constant_memory and row > self.previous_row:
            self._write_single_row(row)
        if cell_format is None:
            cell_format = self.default_url_format
        self._write_string(row, col, string, cell_format)
        self.hyperlinks[row][col] = {'link_type': link_type, 'url': url, 'str': url_str, 'tip': tip}
        return str_error

    @convert_cell_args
    def write_rich_string(self, row, col, *args):
        if False:
            print('Hello World!')
        '\n        Write a "rich" string with multiple formats to a worksheet cell.\n\n        Args:\n            row:          The cell row (zero indexed).\n            col:          The cell column (zero indexed).\n            string_parts: String and format pairs.\n            cell_format:  Optional Format object.\n\n        Returns:\n            0:  Success.\n            -1: Row or column is out of worksheet bounds.\n            -2: String truncated to 32k characters.\n            -3: 2 consecutive formats used.\n            -4: Empty string used.\n            -5: Insufficient parameters.\n\n        '
        return self._write_rich_string(row, col, *args)

    def _write_rich_string(self, row, col, *args):
        if False:
            while True:
                i = 10
        tokens = list(args)
        cell_format = None
        string_index = 0
        raw_string = ''
        if self._check_dimensions(row, col):
            return -1
        if isinstance(tokens[-1], Format):
            cell_format = tokens.pop()
        fh = StringIO()
        self.rstring = XMLwriter()
        self.rstring._set_filehandle(fh)
        default = Format()
        fragments = []
        previous = 'format'
        pos = 0
        if len(tokens) <= 2:
            warn('You must specify more than 2 format/fragments for rich strings. Ignoring input in write_rich_string().')
            return -5
        for token in tokens:
            if not isinstance(token, Format):
                if previous != 'format':
                    fragments.append(default)
                    fragments.append(token)
                else:
                    fragments.append(token)
                if token == '':
                    warn("Excel doesn't allow empty strings in rich strings. Ignoring input in write_rich_string().")
                    return -4
                raw_string += token
                previous = 'string'
            else:
                if previous == 'format' and pos > 0:
                    warn("Excel doesn't allow 2 consecutive formats in rich strings. Ignoring input in write_rich_string().")
                    return -3
                fragments.append(token)
                previous = 'format'
            pos += 1
        if not isinstance(fragments[0], Format):
            self.rstring._xml_start_tag('r')
        for token in fragments:
            if isinstance(token, Format):
                self.rstring._xml_start_tag('r')
                self._write_font(token)
            else:
                attributes = []
                if preserve_whitespace(token):
                    attributes.append(('xml:space', 'preserve'))
                self.rstring._xml_data_element('t', token, attributes)
                self.rstring._xml_end_tag('r')
        string = self.rstring.fh.getvalue()
        if len(raw_string) > self.xls_strmax:
            warn("String length must be less than or equal to Excel's limit of 32,767 characters in write_rich_string().")
            return -2
        if not self.constant_memory:
            string_index = self.str_table._get_shared_string_index(string)
        else:
            string_index = string
        if self.constant_memory and row > self.previous_row:
            self._write_single_row(row)
        self.table[row][col] = cell_rich_string_tuple(string_index, cell_format, raw_string)
        return 0

    def add_write_handler(self, user_type, user_function):
        if False:
            return 10
        '\n        Add a callback function to the write() method to handle user defined\n        types.\n\n        Args:\n            user_type:      The user type() to match on.\n            user_function:  The user defined function to write the type data.\n        Returns:\n            Nothing.\n\n        '
        self.write_handlers[user_type] = user_function

    @convert_cell_args
    def write_row(self, row, col, data, cell_format=None):
        if False:
            return 10
        '\n        Write a row of data starting from (row, col).\n\n        Args:\n            row:    The cell row (zero indexed).\n            col:    The cell column (zero indexed).\n            data:   A list of tokens to be written with write().\n            format: An optional cell Format object.\n        Returns:\n            0:  Success.\n            other: Return value of write() method.\n\n        '
        for token in data:
            error = self._write(row, col, token, cell_format)
            if error:
                return error
            col += 1
        return 0

    @convert_cell_args
    def write_column(self, row, col, data, cell_format=None):
        if False:
            i = 10
            return i + 15
        '\n        Write a column of data starting from (row, col).\n\n        Args:\n            row:    The cell row (zero indexed).\n            col:    The cell column (zero indexed).\n            data:   A list of tokens to be written with write().\n            format: An optional cell Format object.\n        Returns:\n            0:  Success.\n            other: Return value of write() method.\n\n        '
        for token in data:
            error = self._write(row, col, token, cell_format)
            if error:
                return error
            row += 1
        return 0

    @convert_cell_args
    def insert_image(self, row, col, filename, options=None):
        if False:
            print('Hello World!')
        '\n        Insert an image with its top-left corner in a worksheet cell.\n\n        Args:\n            row:      The cell row (zero indexed).\n            col:      The cell column (zero indexed).\n            filename: Path and filename for in supported formats.\n            options:  Position, scale, url and data stream of the image.\n\n        Returns:\n            0:  Success.\n            -1: Row or column is out of worksheet bounds.\n\n        '
        if self._check_dimensions(row, col, True, True):
            warn('Cannot insert image at (%d, %d).' % (row, col))
            return -1
        if options is None:
            options = {}
        x_offset = options.get('x_offset', 0)
        y_offset = options.get('y_offset', 0)
        x_scale = options.get('x_scale', 1)
        y_scale = options.get('y_scale', 1)
        url = options.get('url', None)
        tip = options.get('tip', None)
        anchor = options.get('object_position', 2)
        image_data = options.get('image_data', None)
        description = options.get('description', None)
        decorative = options.get('decorative', False)
        anchor = options.get('positioning', anchor)
        if not image_data and (not os.path.exists(filename)):
            warn("Image file '%s' not found." % filename)
            return -1
        self.images.append([row, col, filename, x_offset, y_offset, x_scale, y_scale, url, tip, anchor, image_data, description, decorative])
        return 0

    @convert_cell_args
    def insert_textbox(self, row, col, text, options=None):
        if False:
            return 10
        '\n        Insert an textbox with its top-left corner in a worksheet cell.\n\n        Args:\n            row:      The cell row (zero indexed).\n            col:      The cell column (zero indexed).\n            text:     The text for the textbox.\n            options:  Textbox options.\n\n        Returns:\n            0:  Success.\n            -1: Row or column is out of worksheet bounds.\n\n        '
        if self._check_dimensions(row, col, True, True):
            warn('Cannot insert textbox at (%d, %d).' % (row, col))
            return -1
        if text is None:
            text = ''
        if options is None:
            options = {}
        x_offset = options.get('x_offset', 0)
        y_offset = options.get('y_offset', 0)
        x_scale = options.get('x_scale', 1)
        y_scale = options.get('y_scale', 1)
        anchor = options.get('object_position', 1)
        description = options.get('description', None)
        decorative = options.get('decorative', False)
        self.shapes.append([row, col, x_offset, y_offset, x_scale, y_scale, text, anchor, options, description, decorative])
        return 0

    @convert_cell_args
    def insert_chart(self, row, col, chart, options=None):
        if False:
            return 10
        '\n        Insert an chart with its top-left corner in a worksheet cell.\n\n        Args:\n            row:     The cell row (zero indexed).\n            col:     The cell column (zero indexed).\n            chart:   Chart object.\n            options: Position and scale of the chart.\n\n        Returns:\n            0:  Success.\n            -1: Row or column is out of worksheet bounds.\n\n        '
        if self._check_dimensions(row, col, True, True):
            warn('Cannot insert chart at (%d, %d).' % (row, col))
            return -1
        if options is None:
            options = {}
        if chart.already_inserted or (chart.combined and chart.combined.already_inserted):
            warn('Chart cannot be inserted in a worksheet more than once.')
            return
        else:
            chart.already_inserted = True
            if chart.combined:
                chart.combined.already_inserted = True
        x_offset = options.get('x_offset', 0)
        y_offset = options.get('y_offset', 0)
        x_scale = options.get('x_scale', 1)
        y_scale = options.get('y_scale', 1)
        anchor = options.get('object_position', 1)
        description = options.get('description', None)
        decorative = options.get('decorative', False)
        if chart.x_scale != 1:
            x_scale = chart.x_scale
        if chart.y_scale != 1:
            y_scale = chart.y_scale
        if chart.x_offset:
            x_offset = chart.x_offset
        if chart.y_offset:
            y_offset = chart.y_offset
        self.charts.append([row, col, chart, x_offset, y_offset, x_scale, y_scale, anchor, description, decorative])
        return 0

    @convert_cell_args
    def write_comment(self, row, col, comment, options=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Write a comment to a worksheet cell.\n\n        Args:\n            row:     The cell row (zero indexed).\n            col:     The cell column (zero indexed).\n            comment: Cell comment. Str.\n            options: Comment formatting options.\n\n        Returns:\n            0:  Success.\n            -1: Row or column is out of worksheet bounds.\n            -2: String longer than 32k characters.\n\n        '
        if options is None:
            options = {}
        if self._check_dimensions(row, col):
            return -1
        if len(comment) > self.xls_strmax:
            return -2
        self.has_vml = 1
        self.has_comments = 1
        self.comments[row][col] = [row, col, comment, options]
        return 0

    def show_comments(self):
        if False:
            return 10
        '\n        Make any comments in the worksheet visible.\n\n        Args:\n            None.\n\n        Returns:\n            Nothing.\n\n        '
        self.comments_visible = 1

    def set_background(self, filename, is_byte_stream=False):
        if False:
            while True:
                i = 10
        '\n        Set a background image for a worksheet.\n\n        Args:\n            filename:       Path and filename for in supported formats.\n            is_byte_stream: File is a stream of bytes.\n        Returns:\n            Nothing.\n\n        '
        if not is_byte_stream and (not os.path.exists(filename)):
            warn("Image file '%s' not found." % filename)
            return -1
        self.background_bytes = is_byte_stream
        self.background_image = filename

    def set_comments_author(self, author):
        if False:
            while True:
                i = 10
        '\n        Set the default author of the cell comments.\n\n        Args:\n            author: Comment author name. String.\n\n        Returns:\n            Nothing.\n\n        '
        self.comments_author = author

    def get_name(self):
        if False:
            while True:
                i = 10
        '\n        Retrieve the worksheet name.\n\n        Args:\n            None.\n\n        Returns:\n            Nothing.\n\n        '
        return self.name

    def activate(self):
        if False:
            print('Hello World!')
        '\n        Set this worksheet as the active worksheet, i.e. the worksheet that is\n        displayed when the workbook is opened. Also set it as selected.\n\n        Note: An active worksheet cannot be hidden.\n\n        Args:\n            None.\n\n        Returns:\n            Nothing.\n\n        '
        self.hidden = 0
        self.selected = 1
        self.worksheet_meta.activesheet = self.index

    def select(self):
        if False:
            while True:
                i = 10
        '\n        Set current worksheet as a selected worksheet, i.e. the worksheet\n        has its tab highlighted.\n\n        Note: A selected worksheet cannot be hidden.\n\n        Args:\n            None.\n\n        Returns:\n            Nothing.\n\n        '
        self.selected = 1
        self.hidden = 0

    def hide(self):
        if False:
            return 10
        '\n        Hide the current worksheet.\n\n        Args:\n            None.\n\n        Returns:\n            Nothing.\n\n        '
        self.hidden = 1
        self.selected = 0

    def very_hidden(self):
        if False:
            print('Hello World!')
        '\n        Hide the current worksheet. This can only be unhidden by VBA.\n\n        Args:\n            None.\n\n        Returns:\n            Nothing.\n\n        '
        self.hidden = 2
        self.selected = 0

    def set_first_sheet(self):
        if False:
            return 10
        '\n        Set current worksheet as the first visible sheet. This is necessary\n        when there are a large number of worksheets and the activated\n        worksheet is not visible on the screen.\n\n        Note: A selected worksheet cannot be hidden.\n\n        Args:\n            None.\n\n        Returns:\n            Nothing.\n\n        '
        self.hidden = 0
        self.worksheet_meta.firstsheet = self.index

    @convert_column_args
    def set_column(self, first_col, last_col, width=None, cell_format=None, options=None):
        if False:
            while True:
                i = 10
        '\n        Set the width, and other properties of a single column or a\n        range of columns.\n\n        Args:\n            first_col:   First column (zero-indexed).\n            last_col:    Last column (zero-indexed). Can be same as first_col.\n            width:       Column width. (optional).\n            cell_format: Column cell_format. (optional).\n            options:     Dict of options such as hidden and level.\n\n        Returns:\n            0:  Success.\n            -1: Column number is out of worksheet bounds.\n\n        '
        if options is None:
            options = {}
        if first_col > last_col:
            (first_col, last_col) = (last_col, first_col)
        ignore_row = True
        hidden = options.get('hidden', False)
        collapsed = options.get('collapsed', False)
        level = options.get('level', 0)
        if cell_format or (width and hidden):
            ignore_col = False
        else:
            ignore_col = True
        if self._check_dimensions(0, last_col, ignore_row, ignore_col):
            return -1
        if self._check_dimensions(0, first_col, ignore_row, ignore_col):
            return -1
        if level < 0:
            level = 0
        if level > 7:
            level = 7
        if level > self.outline_col_level:
            self.outline_col_level = level
        for col in range(first_col, last_col + 1):
            self.col_info[col] = [width, cell_format, hidden, level, collapsed, False]
        self.col_size_changed = True
        return 0

    @convert_column_args
    def set_column_pixels(self, first_col, last_col, width=None, cell_format=None, options=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the width, and other properties of a single column or a\n        range of columns, where column width is in pixels.\n\n        Args:\n            first_col:   First column (zero-indexed).\n            last_col:    Last column (zero-indexed). Can be same as first_col.\n            width:       Column width in pixels. (optional).\n            cell_format: Column cell_format. (optional).\n            options:     Dict of options such as hidden and level.\n\n        Returns:\n            0:  Success.\n            -1: Column number is out of worksheet bounds.\n\n        '
        if width is not None:
            width = self._pixels_to_width(width)
        return self.set_column(first_col, last_col, width, cell_format, options)

    def autofit(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Simulate autofit based on the data, and datatypes in each column.\n\n        Args:\n            None.\n\n        Returns:\n            Nothing.\n\n        '
        if self.constant_memory:
            warn('Autofit is not supported in constant_memory mode.')
            return
        if self.dim_rowmax is None:
            return
        col_width_max = {}
        strings = sorted(self.str_table.string_table, key=self.str_table.string_table.__getitem__)
        for row_num in range(self.dim_rowmin, self.dim_rowmax + 1):
            if not self.table.get(row_num):
                continue
            for col_num in range(self.dim_colmin, self.dim_colmax + 1):
                if col_num in self.table[row_num]:
                    cell = self.table[row_num][col_num]
                    cell_type = cell.__class__.__name__
                    length = 0
                    if cell_type in ('String', 'RichString'):
                        if cell_type == 'String':
                            string_id = cell.string
                            string = strings[string_id]
                        else:
                            string = cell.raw_string
                        if '\n' not in string:
                            length = xl_pixel_width(string)
                        else:
                            for string in string.split('\n'):
                                seg_length = xl_pixel_width(string)
                                if seg_length > length:
                                    length = seg_length
                    elif cell_type == 'Number':
                        length = 7 * len(str(cell.number))
                    elif cell_type == 'Datetime':
                        length = self.default_date_pixels
                    elif cell_type == 'Boolean':
                        if cell.boolean:
                            length = 31
                        else:
                            length = 36
                    elif cell_type == 'Formula' or cell_type == 'ArrayFormula':
                        if isinstance(cell.value, (float, int)):
                            if cell.value > 0:
                                length = 7 * len(str(cell.value))
                        elif isinstance(cell.value, str):
                            length = xl_pixel_width(cell.value)
                        elif isinstance(cell.value, bool):
                            if cell.value:
                                length = 31
                            else:
                                length = 36
                    if self.filter_cells.get((row_num, col_num)) and length > 0:
                        length += 16
                    width_max = col_width_max.get(col_num, 0)
                    if length > width_max:
                        col_width_max[col_num] = length
        for (col_num, pixel_width) in col_width_max.items():
            width = self._pixels_to_width(pixel_width + 7)
            if width > 255.0:
                width = 255.0
            if self.col_info.get(col_num):
                col_info = self.col_info.get(col_num)
                user_width = col_info[0]
                hidden = col_info[5]
                if user_width is not None and (not hidden):
                    if width > user_width:
                        self.col_info[col_num][0] = width
                        self.col_info[col_num][5] = True
                else:
                    self.col_info[col_num][0] = width
                    self.col_info[col_num][5] = True
            else:
                self.col_info[col_num] = [width, None, False, 0, False, True]

    def set_row(self, row, height=None, cell_format=None, options=None):
        if False:
            i = 10
            return i + 15
        '\n        Set the width, and other properties of a row.\n\n        Args:\n            row:         Row number (zero-indexed).\n            height:      Row height. (optional).\n            cell_format: Row cell_format. (optional).\n            options:     Dict of options such as hidden, level and collapsed.\n\n        Returns:\n            0:  Success.\n            -1: Row number is out of worksheet bounds.\n\n        '
        if options is None:
            options = {}
        if self.dim_colmin is not None:
            min_col = self.dim_colmin
        else:
            min_col = 0
        if self._check_dimensions(row, min_col):
            return -1
        if height is None:
            height = self.default_row_height
        hidden = options.get('hidden', False)
        collapsed = options.get('collapsed', False)
        level = options.get('level', 0)
        if height == 0:
            hidden = 1
            height = self.default_row_height
        if level < 0:
            level = 0
        if level > 7:
            level = 7
        if level > self.outline_row_level:
            self.outline_row_level = level
        self.set_rows[row] = [height, cell_format, hidden, level, collapsed]
        self.row_size_changed = True
        self.row_sizes[row] = [height, hidden]
        return 0

    def set_row_pixels(self, row, height=None, cell_format=None, options=None):
        if False:
            i = 10
            return i + 15
        '\n        Set the width (in pixels), and other properties of a row.\n\n        Args:\n            row:         Row number (zero-indexed).\n            height:      Row height in pixels. (optional).\n            cell_format: Row cell_format. (optional).\n            options:     Dict of options such as hidden, level and collapsed.\n\n        Returns:\n            0:  Success.\n            -1: Row number is out of worksheet bounds.\n\n        '
        if height is not None:
            height = self._pixels_to_height(height)
        return self.set_row(row, height, cell_format, options)

    def set_default_row(self, height=None, hide_unused_rows=False):
        if False:
            return 10
        '\n        Set the default row properties.\n\n        Args:\n            height:           Default height. Optional, defaults to 15.\n            hide_unused_rows: Hide unused rows. Optional, defaults to False.\n\n        Returns:\n            Nothing.\n\n        '
        if height is None:
            height = self.default_row_height
        if height != self.original_row_height:
            self.row_size_changed = True
            self.default_row_height = height
        if hide_unused_rows:
            self.default_row_zeroed = 1

    @convert_range_args
    def merge_range(self, first_row, first_col, last_row, last_col, data, cell_format=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Merge a range of cells.\n\n        Args:\n            first_row:    The first row of the cell range. (zero indexed).\n            first_col:    The first column of the cell range.\n            last_row:     The last row of the cell range. (zero indexed).\n            last_col:     The last column of the cell range.\n            data:         Cell data.\n            cell_format:  Cell Format object.\n\n        Returns:\n             0:    Success.\n            -1:    Row or column is out of worksheet bounds.\n            other: Return value of write().\n\n        '
        if first_row == last_row and first_col == last_col:
            warn("Can't merge single cell")
            return
        if first_row > last_row:
            (first_row, last_row) = (last_row, first_row)
        if first_col > last_col:
            (first_col, last_col) = (last_col, first_col)
        if self._check_dimensions(first_row, first_col):
            return -1
        if self._check_dimensions(last_row, last_col):
            return -1
        cell_range = xl_range(first_row, first_col, last_row, last_col)
        for row in range(first_row, last_row + 1):
            for col in range(first_col, last_col + 1):
                if self.merged_cells.get((row, col)):
                    previous_range = self.merged_cells.get((row, col))
                    raise OverlappingRange("Merge range '%s' overlaps previous merge range '%s'." % (cell_range, previous_range))
                elif self.table_cells.get((row, col)):
                    previous_range = self.table_cells.get((row, col))
                    raise OverlappingRange("Merge range '%s' overlaps previous table range '%s'." % (cell_range, previous_range))
                else:
                    self.merged_cells[row, col] = cell_range
        self.merge.append([first_row, first_col, last_row, last_col])
        self._write(first_row, first_col, data, cell_format)
        for row in range(first_row, last_row + 1):
            for col in range(first_col, last_col + 1):
                if row == first_row and col == first_col:
                    continue
                self._write_blank(row, col, '', cell_format)
        return 0

    @convert_range_args
    def autofilter(self, first_row, first_col, last_row, last_col):
        if False:
            i = 10
            return i + 15
        '\n        Set the autofilter area in the worksheet.\n\n        Args:\n            first_row:    The first row of the cell range. (zero indexed).\n            first_col:    The first column of the cell range.\n            last_row:     The last row of the cell range. (zero indexed).\n            last_col:     The last column of the cell range.\n\n        Returns:\n             Nothing.\n\n        '
        if last_row < first_row:
            (first_row, last_row) = (last_row, first_row)
        if last_col < first_col:
            (first_col, last_col) = (last_col, first_col)
        area = self._convert_name_area(first_row, first_col, last_row, last_col)
        ref = xl_range(first_row, first_col, last_row, last_col)
        self.autofilter_area = area
        self.autofilter_ref = ref
        self.filter_range = [first_col, last_col]
        for col in range(first_col, last_col + 1):
            self.filter_cells[first_row, col] = True

    def filter_column(self, col, criteria):
        if False:
            while True:
                i = 10
        '\n        Set the column filter criteria.\n\n        Args:\n            col:       Filter column (zero-indexed).\n            criteria:  Filter criteria.\n\n        Returns:\n             Nothing.\n\n        '
        if not self.autofilter_area:
            warn('Must call autofilter() before filter_column()')
            return
        try:
            int(col)
        except ValueError:
            col_letter = col
            (_, col) = xl_cell_to_rowcol(col + '1')
            if col >= self.xls_colmax:
                warn("Invalid column '%s'" % col_letter)
                return
        (col_first, col_last) = self.filter_range
        if col < col_first or col > col_last:
            warn("Column '%d' outside autofilter() column range (%d, %d)" % (col, col_first, col_last))
            return
        tokens = self._extract_filter_tokens(criteria)
        if len(tokens) not in (3, 7):
            warn("Incorrect number of tokens in criteria '%s'" % criteria)
        tokens = self._parse_filter_expression(criteria, tokens)
        if len(tokens) == 2 and tokens[0] == 2:
            self.filter_column_list(col, [tokens[1]])
        elif len(tokens) == 5 and tokens[0] == 2 and (tokens[2] == 1) and (tokens[3] == 2):
            self.filter_column_list(col, [tokens[1], tokens[4]])
        else:
            self.filter_cols[col] = tokens
            self.filter_type[col] = 0
        self.filter_on = 1

    def filter_column_list(self, col, filters):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the column filter criteria in Excel 2007 list style.\n\n        Args:\n            col:      Filter column (zero-indexed).\n            filters:  List of filter criteria to match.\n\n        Returns:\n             Nothing.\n\n        '
        if not self.autofilter_area:
            warn('Must call autofilter() before filter_column()')
            return
        try:
            int(col)
        except ValueError:
            col_letter = col
            (_, col) = xl_cell_to_rowcol(col + '1')
            if col >= self.xls_colmax:
                warn("Invalid column '%s'" % col_letter)
                return
        (col_first, col_last) = self.filter_range
        if col < col_first or col > col_last:
            warn("Column '%d' outside autofilter() column range (%d,%d)" % (col, col_first, col_last))
            return
        self.filter_cols[col] = filters
        self.filter_type[col] = 1
        self.filter_on = 1

    @convert_range_args
    def data_validation(self, first_row, first_col, last_row, last_col, options=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add a data validation to a worksheet.\n\n        Args:\n            first_row:    The first row of the cell range. (zero indexed).\n            first_col:    The first column of the cell range.\n            last_row:     The last row of the cell range. (zero indexed).\n            last_col:     The last column of the cell range.\n            options:      Data validation options.\n\n        Returns:\n            0:  Success.\n            -1: Row or column is out of worksheet bounds.\n            -2: Incorrect parameter or option.\n        '
        if self._check_dimensions(first_row, first_col, True, True):
            return -1
        if self._check_dimensions(last_row, last_col, True, True):
            return -1
        if options is None:
            options = {}
        else:
            options = options.copy()
        valid_parameters = {'validate', 'criteria', 'value', 'source', 'minimum', 'maximum', 'ignore_blank', 'dropdown', 'show_input', 'input_title', 'input_message', 'show_error', 'error_title', 'error_message', 'error_type', 'other_cells', 'multi_range'}
        for param_key in options.keys():
            if param_key not in valid_parameters:
                warn("Unknown parameter '%s' in data_validation()" % param_key)
                return -2
        if 'source' in options:
            options['value'] = options['source']
        if 'minimum' in options:
            options['value'] = options['minimum']
        if 'validate' not in options:
            warn("Parameter 'validate' is required in data_validation()")
            return -2
        valid_types = {'any': 'none', 'any value': 'none', 'whole number': 'whole', 'whole': 'whole', 'integer': 'whole', 'decimal': 'decimal', 'list': 'list', 'date': 'date', 'time': 'time', 'text length': 'textLength', 'length': 'textLength', 'custom': 'custom'}
        if options['validate'] not in valid_types:
            warn("Unknown validation type '%s' for parameter 'validate' in data_validation()" % options['validate'])
            return -2
        else:
            options['validate'] = valid_types[options['validate']]
        if options['validate'] == 'none' and options.get('input_title') is None and (options.get('input_message') is None):
            return -2
        if options['validate'] == 'none' or options['validate'] == 'list' or options['validate'] == 'custom':
            options['criteria'] = 'between'
            options['maximum'] = None
        if 'criteria' not in options:
            warn("Parameter 'criteria' is required in data_validation()")
            return -2
        criteria_types = {'between': 'between', 'not between': 'notBetween', 'equal to': 'equal', '=': 'equal', '==': 'equal', 'not equal to': 'notEqual', '!=': 'notEqual', '<>': 'notEqual', 'greater than': 'greaterThan', '>': 'greaterThan', 'less than': 'lessThan', '<': 'lessThan', 'greater than or equal to': 'greaterThanOrEqual', '>=': 'greaterThanOrEqual', 'less than or equal to': 'lessThanOrEqual', '<=': 'lessThanOrEqual'}
        if options['criteria'] not in criteria_types:
            warn("Unknown criteria type '%s' for parameter 'criteria' in data_validation()" % options['criteria'])
            return -2
        else:
            options['criteria'] = criteria_types[options['criteria']]
        if options['criteria'] == 'between' or options['criteria'] == 'notBetween':
            if 'maximum' not in options:
                warn("Parameter 'maximum' is required in data_validation() when using 'between' or 'not between' criteria")
                return -2
        else:
            options['maximum'] = None
        error_types = {'stop': 0, 'warning': 1, 'information': 2}
        if 'error_type' not in options:
            options['error_type'] = 0
        elif options['error_type'] not in error_types:
            warn("Unknown criteria type '%s' for parameter 'error_type' in data_validation()" % options['error_type'])
            return -2
        else:
            options['error_type'] = error_types[options['error_type']]
        if options['validate'] in ('date', 'time') and options['value'] and supported_datetime(options['value']):
            date_time = self._convert_date_time(options['value'])
            options['value'] = '%.16g' % date_time
            if options['maximum'] and supported_datetime(options['maximum']):
                date_time = self._convert_date_time(options['maximum'])
                options['maximum'] = '%.16g' % date_time
        if options.get('input_title') and len(options['input_title']) > 32:
            warn("Length of input title '%s' exceeds Excel's limit of 32" % options['input_title'])
            return -2
        if options.get('error_title') and len(options['error_title']) > 32:
            warn("Length of error title '%s' exceeds Excel's limit of 32" % options['error_title'])
            return -2
        if options.get('input_message') and len(options['input_message']) > 255:
            warn("Length of input message '%s' exceeds Excel's limit of 255" % options['input_message'])
            return -2
        if options.get('error_message') and len(options['error_message']) > 255:
            warn("Length of error message '%s' exceeds Excel's limit of 255" % options['error_message'])
            return -2
        if options['validate'] == 'list' and isinstance(options['value'], list):
            formula = self._csv_join(*options['value'])
            if len(formula) > 255:
                warn("Length of list items '%s' exceeds Excel's limit of 255, use a formula range instead" % formula)
                return -2
        if 'ignore_blank' not in options:
            options['ignore_blank'] = 1
        if 'dropdown' not in options:
            options['dropdown'] = 1
        if 'show_input' not in options:
            options['show_input'] = 1
        if 'show_error' not in options:
            options['show_error'] = 1
        options['cells'] = [[first_row, first_col, last_row, last_col]]
        if 'other_cells' in options:
            options['cells'].extend(options['other_cells'])
        if 'multi_range' in options:
            options['multi_range'] = options['multi_range'].replace('$', '')
        self.validations.append(options)
        return 0

    @convert_range_args
    def conditional_format(self, first_row, first_col, last_row, last_col, options=None):
        if False:
            i = 10
            return i + 15
        '\n        Add a conditional format to a worksheet.\n\n        Args:\n            first_row:    The first row of the cell range. (zero indexed).\n            first_col:    The first column of the cell range.\n            last_row:     The last row of the cell range. (zero indexed).\n            last_col:     The last column of the cell range.\n            options:      Conditional format options.\n\n        Returns:\n            0:  Success.\n            -1: Row or column is out of worksheet bounds.\n            -2: Incorrect parameter or option.\n        '
        if self._check_dimensions(first_row, first_col, True, True):
            return -1
        if self._check_dimensions(last_row, last_col, True, True):
            return -1
        if options is None:
            options = {}
        else:
            options = options.copy()
        valid_parameter = {'type', 'format', 'criteria', 'value', 'minimum', 'maximum', 'stop_if_true', 'min_type', 'mid_type', 'max_type', 'min_value', 'mid_value', 'max_value', 'min_color', 'mid_color', 'max_color', 'min_length', 'max_length', 'multi_range', 'bar_color', 'bar_negative_color', 'bar_negative_color_same', 'bar_solid', 'bar_border_color', 'bar_negative_border_color', 'bar_negative_border_color_same', 'bar_no_border', 'bar_direction', 'bar_axis_position', 'bar_axis_color', 'bar_only', 'data_bar_2010', 'icon_style', 'reverse_icons', 'icons_only', 'icons'}
        for param_key in options.keys():
            if param_key not in valid_parameter:
                warn("Unknown parameter '%s' in conditional_format()" % param_key)
                return -2
        if 'type' not in options:
            warn("Parameter 'type' is required in conditional_format()")
            return -2
        valid_type = {'cell': 'cellIs', 'date': 'date', 'time': 'time', 'average': 'aboveAverage', 'duplicate': 'duplicateValues', 'unique': 'uniqueValues', 'top': 'top10', 'bottom': 'top10', 'text': 'text', 'time_period': 'timePeriod', 'blanks': 'containsBlanks', 'no_blanks': 'notContainsBlanks', 'errors': 'containsErrors', 'no_errors': 'notContainsErrors', '2_color_scale': '2_color_scale', '3_color_scale': '3_color_scale', 'data_bar': 'dataBar', 'formula': 'expression', 'icon_set': 'iconSet'}
        if options['type'] not in valid_type:
            warn("Unknown value '%s' for parameter 'type' in conditional_format()" % options['type'])
            return -2
        else:
            if options['type'] == 'bottom':
                options['direction'] = 'bottom'
            options['type'] = valid_type[options['type']]
        criteria_type = {'between': 'between', 'not between': 'notBetween', 'equal to': 'equal', '=': 'equal', '==': 'equal', 'not equal to': 'notEqual', '!=': 'notEqual', '<>': 'notEqual', 'greater than': 'greaterThan', '>': 'greaterThan', 'less than': 'lessThan', '<': 'lessThan', 'greater than or equal to': 'greaterThanOrEqual', '>=': 'greaterThanOrEqual', 'less than or equal to': 'lessThanOrEqual', '<=': 'lessThanOrEqual', 'containing': 'containsText', 'not containing': 'notContains', 'begins with': 'beginsWith', 'ends with': 'endsWith', 'yesterday': 'yesterday', 'today': 'today', 'last 7 days': 'last7Days', 'last week': 'lastWeek', 'this week': 'thisWeek', 'next week': 'nextWeek', 'last month': 'lastMonth', 'this month': 'thisMonth', 'next month': 'nextMonth', 'continue week': 'nextWeek', 'continue month': 'nextMonth'}
        if 'criteria' in options and options['criteria'] in criteria_type:
            options['criteria'] = criteria_type[options['criteria']]
        if options['type'] in ('date', 'time'):
            options['type'] = 'cellIs'
            if 'value' in options:
                if not supported_datetime(options['value']):
                    warn("Conditional format 'value' must be a datetime object.")
                    return -2
                else:
                    date_time = self._convert_date_time(options['value'])
                    options['value'] = '%.16g' % date_time
            if 'minimum' in options:
                if not supported_datetime(options['minimum']):
                    warn("Conditional format 'minimum' must be a datetime object.")
                    return -2
                else:
                    date_time = self._convert_date_time(options['minimum'])
                    options['minimum'] = '%.16g' % date_time
            if 'maximum' in options:
                if not supported_datetime(options['maximum']):
                    warn("Conditional format 'maximum' must be a datetime object.")
                    return -2
                else:
                    date_time = self._convert_date_time(options['maximum'])
                    options['maximum'] = '%.16g' % date_time
        valid_icons = {'3_arrows': '3Arrows', '3_flags': '3Flags', '3_traffic_lights_rimmed': '3TrafficLights2', '3_symbols_circled': '3Symbols', '4_arrows': '4Arrows', '4_red_to_black': '4RedToBlack', '4_traffic_lights': '4TrafficLights', '5_arrows_gray': '5ArrowsGray', '5_quarters': '5Quarters', '3_arrows_gray': '3ArrowsGray', '3_traffic_lights': '3TrafficLights', '3_signs': '3Signs', '3_symbols': '3Symbols2', '4_arrows_gray': '4ArrowsGray', '4_ratings': '4Rating', '5_arrows': '5Arrows', '5_ratings': '5Rating'}
        if options['type'] == 'iconSet':
            if not options.get('icon_style'):
                warn("The 'icon_style' parameter must be specified when 'type' == 'icon_set' in conditional_format()")
                return -3
            if options['icon_style'] not in valid_icons:
                warn("Unknown icon_style '%s' in conditional_format()" % options['icon_style'])
                return -2
            else:
                options['icon_style'] = valid_icons[options['icon_style']]
            options['total_icons'] = 3
            if options['icon_style'].startswith('4'):
                options['total_icons'] = 4
            elif options['icon_style'].startswith('5'):
                options['total_icons'] = 5
            options['icons'] = self._set_icon_props(options.get('total_icons'), options.get('icons'))
        if first_row > last_row:
            (first_row, last_row) = (last_row, first_row)
        if first_col > last_col:
            (first_col, last_col) = (last_col, first_col)
        cell_range = xl_range(first_row, first_col, last_row, last_col)
        start_cell = xl_rowcol_to_cell(first_row, first_col)
        if 'multi_range' in options:
            cell_range = options['multi_range']
            cell_range = cell_range.replace('$', '')
        if 'format' in options and options['format']:
            options['format'] = options['format']._get_dxf_index()
        options['priority'] = self.dxf_priority
        self.dxf_priority += 1
        if self.use_data_bars_2010 or options.get('data_bar_2010') or options.get('bar_solid') or options.get('bar_border_color') or options.get('bar_negative_color') or options.get('bar_negative_color_same') or options.get('bar_negative_border_color') or options.get('bar_negative_border_color_same') or options.get('bar_no_border') or options.get('bar_axis_position') or options.get('bar_axis_color') or options.get('bar_direction'):
            options['is_data_bar_2010'] = True
        if options['type'] == 'text':
            if options['criteria'] == 'containsText':
                options['type'] = 'containsText'
                options['formula'] = 'NOT(ISERROR(SEARCH("%s",%s)))' % (options['value'], start_cell)
            elif options['criteria'] == 'notContains':
                options['type'] = 'notContainsText'
                options['formula'] = 'ISERROR(SEARCH("%s",%s))' % (options['value'], start_cell)
            elif options['criteria'] == 'beginsWith':
                options['type'] = 'beginsWith'
                options['formula'] = 'LEFT(%s,%d)="%s"' % (start_cell, len(options['value']), options['value'])
            elif options['criteria'] == 'endsWith':
                options['type'] = 'endsWith'
                options['formula'] = 'RIGHT(%s,%d)="%s"' % (start_cell, len(options['value']), options['value'])
            else:
                warn("Invalid text criteria '%s' in conditional_format()" % options['criteria'])
        if options['type'] == 'timePeriod':
            if options['criteria'] == 'yesterday':
                options['formula'] = 'FLOOR(%s,1)=TODAY()-1' % start_cell
            elif options['criteria'] == 'today':
                options['formula'] = 'FLOOR(%s,1)=TODAY()' % start_cell
            elif options['criteria'] == 'tomorrow':
                options['formula'] = 'FLOOR(%s,1)=TODAY()+1' % start_cell
            elif options['criteria'] == 'last7Days':
                options['formula'] = 'AND(TODAY()-FLOOR(%s,1)<=6,FLOOR(%s,1)<=TODAY())' % (start_cell, start_cell)
            elif options['criteria'] == 'lastWeek':
                options['formula'] = 'AND(TODAY()-ROUNDDOWN(%s,0)>=(WEEKDAY(TODAY())),TODAY()-ROUNDDOWN(%s,0)<(WEEKDAY(TODAY())+7))' % (start_cell, start_cell)
            elif options['criteria'] == 'thisWeek':
                options['formula'] = 'AND(TODAY()-ROUNDDOWN(%s,0)<=WEEKDAY(TODAY())-1,ROUNDDOWN(%s,0)-TODAY()<=7-WEEKDAY(TODAY()))' % (start_cell, start_cell)
            elif options['criteria'] == 'nextWeek':
                options['formula'] = 'AND(ROUNDDOWN(%s,0)-TODAY()>(7-WEEKDAY(TODAY())),ROUNDDOWN(%s,0)-TODAY()<(15-WEEKDAY(TODAY())))' % (start_cell, start_cell)
            elif options['criteria'] == 'lastMonth':
                options['formula'] = 'AND(MONTH(%s)=MONTH(TODAY())-1,OR(YEAR(%s)=YEAR(TODAY()),AND(MONTH(%s)=1,YEAR(A1)=YEAR(TODAY())-1)))' % (start_cell, start_cell, start_cell)
            elif options['criteria'] == 'thisMonth':
                options['formula'] = 'AND(MONTH(%s)=MONTH(TODAY()),YEAR(%s)=YEAR(TODAY()))' % (start_cell, start_cell)
            elif options['criteria'] == 'nextMonth':
                options['formula'] = 'AND(MONTH(%s)=MONTH(TODAY())+1,OR(YEAR(%s)=YEAR(TODAY()),AND(MONTH(%s)=12,YEAR(%s)=YEAR(TODAY())+1)))' % (start_cell, start_cell, start_cell, start_cell)
            else:
                warn("Invalid time_period criteria '%s' in conditional_format()" % options['criteria'])
        if options['type'] == 'containsBlanks':
            options['formula'] = 'LEN(TRIM(%s))=0' % start_cell
        if options['type'] == 'notContainsBlanks':
            options['formula'] = 'LEN(TRIM(%s))>0' % start_cell
        if options['type'] == 'containsErrors':
            options['formula'] = 'ISERROR(%s)' % start_cell
        if options['type'] == 'notContainsErrors':
            options['formula'] = 'NOT(ISERROR(%s))' % start_cell
        if options['type'] == '2_color_scale':
            options['type'] = 'colorScale'
            options['format'] = None
            options['mid_type'] = None
            options['mid_color'] = None
            options.setdefault('min_type', 'min')
            options.setdefault('max_type', 'max')
            options.setdefault('min_value', 0)
            options.setdefault('max_value', 0)
            options.setdefault('min_color', '#FF7128')
            options.setdefault('max_color', '#FFEF9C')
            options['min_color'] = xl_color(options['min_color'])
            options['max_color'] = xl_color(options['max_color'])
        if options['type'] == '3_color_scale':
            options['type'] = 'colorScale'
            options['format'] = None
            options.setdefault('min_type', 'min')
            options.setdefault('mid_type', 'percentile')
            options.setdefault('max_type', 'max')
            options.setdefault('min_value', 0)
            options.setdefault('max_value', 0)
            options.setdefault('min_color', '#F8696B')
            options.setdefault('mid_color', '#FFEB84')
            options.setdefault('max_color', '#63BE7B')
            options['min_color'] = xl_color(options['min_color'])
            options['mid_color'] = xl_color(options['mid_color'])
            options['max_color'] = xl_color(options['max_color'])
            if 'mid_value' not in options:
                options['mid_value'] = 50
        if options['type'] == 'dataBar':
            options['format'] = None
            if not options.get('min_type'):
                options['min_type'] = 'min'
                options['x14_min_type'] = 'autoMin'
            else:
                options['x14_min_type'] = options['min_type']
            if not options.get('max_type'):
                options['max_type'] = 'max'
                options['x14_max_type'] = 'autoMax'
            else:
                options['x14_max_type'] = options['max_type']
            options.setdefault('min_value', 0)
            options.setdefault('max_value', 0)
            options.setdefault('bar_color', '#638EC6')
            options.setdefault('bar_border_color', options['bar_color'])
            options.setdefault('bar_only', False)
            options.setdefault('bar_no_border', False)
            options.setdefault('bar_solid', False)
            options.setdefault('bar_direction', '')
            options.setdefault('bar_negative_color', '#FF0000')
            options.setdefault('bar_negative_border_color', '#FF0000')
            options.setdefault('bar_negative_color_same', False)
            options.setdefault('bar_negative_border_color_same', False)
            options.setdefault('bar_axis_position', '')
            options.setdefault('bar_axis_color', '#000000')
            options['bar_color'] = xl_color(options['bar_color'])
            options['bar_border_color'] = xl_color(options['bar_border_color'])
            options['bar_axis_color'] = xl_color(options['bar_axis_color'])
            options['bar_negative_color'] = xl_color(options['bar_negative_color'])
            options['bar_negative_border_color'] = xl_color(options['bar_negative_border_color'])
        if options.get('is_data_bar_2010'):
            self.excel_version = 2010
            if options['min_type'] == 'min' and options['min_value'] == 0:
                options['min_value'] = None
            if options['max_type'] == 'max' and options['max_value'] == 0:
                options['max_value'] = None
            options['range'] = cell_range
        try:
            options['min_value'] = options['min_value'].lstrip('=')
        except (KeyError, AttributeError):
            pass
        try:
            options['mid_value'] = options['mid_value'].lstrip('=')
        except (KeyError, AttributeError):
            pass
        try:
            options['max_value'] = options['max_value'].lstrip('=')
        except (KeyError, AttributeError):
            pass
        if cell_range in self.cond_formats:
            self.cond_formats[cell_range].append(options)
        else:
            self.cond_formats[cell_range] = [options]
        return 0

    @convert_range_args
    def add_table(self, first_row, first_col, last_row, last_col, options=None):
        if False:
            while True:
                i = 10
        '\n        Add an Excel table to a worksheet.\n\n        Args:\n            first_row:    The first row of the cell range. (zero indexed).\n            first_col:    The first column of the cell range.\n            last_row:     The last row of the cell range. (zero indexed).\n            last_col:     The last column of the cell range.\n            options:      Table format options. (Optional)\n\n        Returns:\n            0:  Success.\n            -1: Row or column is out of worksheet bounds.\n            -2: Incorrect parameter or option.\n            -3: Not supported in constant_memory mode.\n        '
        table = {}
        col_formats = {}
        if options is None:
            options = {}
        else:
            options = options.copy()
        if self.constant_memory:
            warn("add_table() isn't supported in 'constant_memory' mode")
            return -3
        if self._check_dimensions(first_row, first_col, True, True):
            return -1
        if self._check_dimensions(last_row, last_col, True, True):
            return -1
        if first_row > last_row:
            (first_row, last_row) = (last_row, first_row)
        if first_col > last_col:
            (first_col, last_col) = (last_col, first_col)
        cell_range = xl_range(first_row, first_col, last_row, last_col)
        for row in range(first_row, last_row + 1):
            for col in range(first_col, last_col + 1):
                if self.table_cells.get((row, col)):
                    previous_range = self.table_cells.get((row, col))
                    raise OverlappingRange("Table range '%s' overlaps previous table range '%s'." % (cell_range, previous_range))
                elif self.merged_cells.get((row, col)):
                    previous_range = self.merged_cells.get((row, col))
                    raise OverlappingRange("Table range '%s' overlaps previous merge range '%s'." % (cell_range, previous_range))
                else:
                    self.table_cells[row, col] = cell_range
        valid_parameter = {'autofilter', 'banded_columns', 'banded_rows', 'columns', 'data', 'first_column', 'header_row', 'last_column', 'name', 'style', 'total_row'}
        for param_key in options.keys():
            if param_key not in valid_parameter:
                warn("Unknown parameter '%s' in add_table()" % param_key)
                return -2
        options['banded_rows'] = options.get('banded_rows', True)
        options['header_row'] = options.get('header_row', True)
        options['autofilter'] = options.get('autofilter', True)
        num_rows = last_row - first_row
        if options['header_row']:
            num_rows -= 1
        if num_rows < 0:
            warn('Must have at least one data row in in add_table()')
            return -2
        table['show_first_col'] = options.get('first_column', False)
        table['show_last_col'] = options.get('last_column', False)
        table['show_row_stripes'] = options.get('banded_rows', False)
        table['show_col_stripes'] = options.get('banded_columns', False)
        table['header_row_count'] = options.get('header_row', 0)
        table['totals_row_shown'] = options.get('total_row', False)
        if 'name' in options:
            name = options['name']
            table['name'] = name
            if ' ' in name:
                warn("Name '%s' in add_table() cannot contain spaces" % name)
                return -2
            if not re.match('^[\\w\\\\][\\w\\\\.]*$', name, re.UNICODE) or re.match('^\\d', name):
                warn("Invalid Excel characters in add_table(): '%s'" % name)
                return -2
            if re.match('^[a-zA-Z][a-zA-Z]?[a-dA-D]?\\d+$', name):
                warn("Name looks like a cell name in add_table(): '%s'" % name)
                return -2
            if re.match('^[rcRC]$', name) or re.match('^[rcRC]\\d+[rcRC]\\d+$', name):
                warn("Invalid name '%s' like a RC cell ref in add_table()" % name)
                return -2
        if 'style' in options:
            table['style'] = options['style']
            if table['style'] is None:
                table['style'] = ''
            table['style'] = table['style'].replace(' ', '')
        else:
            table['style'] = 'TableStyleMedium9'
        first_data_row = first_row
        last_data_row = last_row
        if options.get('header_row'):
            first_data_row += 1
        if options.get('total_row'):
            last_data_row -= 1
        table['range'] = xl_range(first_row, first_col, last_row, last_col)
        table['a_range'] = xl_range(first_row, first_col, last_data_row, last_col)
        if not options['header_row']:
            options['autofilter'] = 0
        if options['autofilter']:
            table['autofilter'] = table['a_range']
        col_id = 1
        table['columns'] = []
        seen_names = {}
        for col_num in range(first_col, last_col + 1):
            col_data = {'id': col_id, 'name': 'Column' + str(col_id), 'total_string': '', 'total_function': '', 'custom_total': '', 'total_value': 0, 'formula': '', 'format': None, 'name_format': None}
            if 'columns' in options:
                if col_id <= len(options['columns']):
                    user_data = options['columns'][col_id - 1]
                else:
                    user_data = None
                if user_data:
                    xformat = user_data.get('format', None)
                    if user_data.get('header'):
                        col_data['name'] = user_data['header']
                    header_name = col_data['name']
                    name = header_name.lower()
                    if name in seen_names:
                        warn("Duplicate header name in add_table(): '%s'" % name)
                        return -2
                    else:
                        seen_names[name] = True
                    col_data['name_format'] = user_data.get('header_format')
                    if 'formula' in user_data and user_data['formula']:
                        formula = user_data['formula']
                        if formula.startswith('='):
                            formula = formula.lstrip('=')
                        formula = formula.replace('@', '[#This Row],')
                        formula = self._prepare_formula(formula, True)
                        col_data['formula'] = formula
                    if user_data.get('total_function'):
                        function = user_data['total_function']
                        if function == 'count_nums':
                            function = 'countNums'
                        if function == 'std_dev':
                            function = 'stdDev'
                        subtotals = set(['average', 'countNums', 'count', 'max', 'min', 'stdDev', 'sum', 'var'])
                        if function in subtotals:
                            formula = self._table_function_to_formula(function, col_data['name'])
                        else:
                            formula = self._prepare_formula(function, True)
                            col_data['custom_total'] = formula
                            function = 'custom'
                        col_data['total_function'] = function
                        value = user_data.get('total_value', 0)
                        self._write_formula(last_row, col_num, formula, xformat, value)
                    elif user_data.get('total_string'):
                        total_string = user_data['total_string']
                        col_data['total_string'] = total_string
                        self._write_string(last_row, col_num, total_string, user_data.get('format'))
                    if xformat is not None:
                        col_data['format'] = xformat._get_dxf_index()
                    col_formats[col_id - 1] = xformat
            table['columns'].append(col_data)
            if options['header_row']:
                self._write_string(first_row, col_num, col_data['name'], col_data['name_format'])
            col_id += 1
        if 'data' in options:
            data = options['data']
            i = 0
            for row in range(first_data_row, last_data_row + 1):
                j = 0
                for col in range(first_col, last_col + 1):
                    if i < len(data) and j < len(data[i]):
                        token = data[i][j]
                        if j in col_formats:
                            self._write(row, col, token, col_formats[j])
                        else:
                            self._write(row, col, token, None)
                    j += 1
                i += 1
        for (col_id, col_num) in enumerate(range(first_col, last_col + 1)):
            column_data = table['columns'][col_id]
            if column_data and column_data['formula']:
                formula_format = col_formats.get(col_id)
                formula = column_data['formula']
                for row in range(first_data_row, last_data_row + 1):
                    self._write_formula(row, col_num, formula, formula_format)
        self.tables.append(table)
        if options['autofilter']:
            for col in range(first_col, last_col + 1):
                self.filter_cells[first_row, col] = True
        return 0

    @convert_cell_args
    def add_sparkline(self, row, col, options=None):
        if False:
            print('Hello World!')
        '\n        Add sparklines to the worksheet.\n\n        Args:\n            row:     The cell row (zero indexed).\n            col:     The cell column (zero indexed).\n            options: Sparkline formatting options.\n\n        Returns:\n            0:  Success.\n            -1: Row or column is out of worksheet bounds.\n            -2: Incorrect parameter or option.\n\n        '
        if self._check_dimensions(row, col, True, True):
            return -1
        sparkline = {'locations': [xl_rowcol_to_cell(row, col)]}
        if options is None:
            options = {}
        valid_parameters = {'location', 'range', 'type', 'high_point', 'low_point', 'negative_points', 'first_point', 'last_point', 'markers', 'style', 'series_color', 'negative_color', 'markers_color', 'first_color', 'last_color', 'high_color', 'low_color', 'max', 'min', 'axis', 'reverse', 'empty_cells', 'show_hidden', 'plot_hidden', 'date_axis', 'weight'}
        for param_key in options.keys():
            if param_key not in valid_parameters:
                warn("Unknown parameter '%s' in add_sparkline()" % param_key)
                return -1
        if 'range' not in options:
            warn("Parameter 'range' is required in add_sparkline()")
            return -2
        spark_type = options.get('type', 'line')
        if spark_type not in ('line', 'column', 'win_loss'):
            warn("Parameter 'type' must be 'line', 'column' or 'win_loss' in add_sparkline()")
            return -2
        if spark_type == 'win_loss':
            spark_type = 'stacked'
        sparkline['type'] = spark_type
        if 'location' in options:
            if isinstance(options['location'], list):
                sparkline['locations'] = options['location']
            else:
                sparkline['locations'] = [options['location']]
        if isinstance(options['range'], list):
            sparkline['ranges'] = options['range']
        else:
            sparkline['ranges'] = [options['range']]
        range_count = len(sparkline['ranges'])
        location_count = len(sparkline['locations'])
        if range_count != location_count:
            warn('Must have the same number of location and range parameters in add_sparkline()')
            return -2
        sparkline['count'] = len(sparkline['locations'])
        sheetname = quote_sheetname(self.name)
        new_ranges = []
        for spark_range in sparkline['ranges']:
            spark_range = spark_range.replace('$', '')
            spark_range = spark_range.lstrip('=')
            if '!' not in spark_range:
                spark_range = sheetname + '!' + spark_range
            new_ranges.append(spark_range)
        sparkline['ranges'] = new_ranges
        new_locations = []
        for location in sparkline['locations']:
            location = location.replace('$', '')
            new_locations.append(location)
        sparkline['locations'] = new_locations
        sparkline['high'] = options.get('high_point')
        sparkline['low'] = options.get('low_point')
        sparkline['negative'] = options.get('negative_points')
        sparkline['first'] = options.get('first_point')
        sparkline['last'] = options.get('last_point')
        sparkline['markers'] = options.get('markers')
        sparkline['min'] = options.get('min')
        sparkline['max'] = options.get('max')
        sparkline['axis'] = options.get('axis')
        sparkline['reverse'] = options.get('reverse')
        sparkline['hidden'] = options.get('show_hidden')
        sparkline['weight'] = options.get('weight')
        empty = options.get('empty_cells', '')
        if empty == 'zero':
            sparkline['empty'] = 0
        elif empty == 'connect':
            sparkline['empty'] = 'span'
        else:
            sparkline['empty'] = 'gap'
        date_range = options.get('date_axis')
        if date_range and '!' not in date_range:
            date_range = sheetname + '!' + date_range
        sparkline['date_axis'] = date_range
        style_id = options.get('style', 0)
        style = get_sparkline_style(style_id)
        sparkline['series_color'] = style['series']
        sparkline['negative_color'] = style['negative']
        sparkline['markers_color'] = style['markers']
        sparkline['first_color'] = style['first']
        sparkline['last_color'] = style['last']
        sparkline['high_color'] = style['high']
        sparkline['low_color'] = style['low']
        self._set_spark_color(sparkline, options, 'series_color')
        self._set_spark_color(sparkline, options, 'negative_color')
        self._set_spark_color(sparkline, options, 'markers_color')
        self._set_spark_color(sparkline, options, 'first_color')
        self._set_spark_color(sparkline, options, 'last_color')
        self._set_spark_color(sparkline, options, 'high_color')
        self._set_spark_color(sparkline, options, 'low_color')
        self.sparklines.append(sparkline)
        return 0

    @convert_range_args
    def set_selection(self, first_row, first_col, last_row, last_col):
        if False:
            i = 10
            return i + 15
        '\n        Set the selected cell or cells in a worksheet\n\n        Args:\n            first_row:    The first row of the cell range. (zero indexed).\n            first_col:    The first column of the cell range.\n            last_row:     The last row of the cell range. (zero indexed).\n            last_col:     The last column of the cell range.\n\n        Returns:\n            0:  Nothing.\n        '
        pane = None
        active_cell = xl_rowcol_to_cell(first_row, first_col)
        if first_row > last_row:
            (first_row, last_row) = (last_row, first_row)
        if first_col > last_col:
            (first_col, last_col) = (last_col, first_col)
        sqref = xl_range(first_row, first_col, last_row, last_col)
        if sqref == 'A1':
            return
        self.selections = [[pane, active_cell, sqref]]

    @convert_cell_args
    def set_top_left_cell(self, row=0, col=0):
        if False:
            while True:
                i = 10
        '\n        Set the first visible cell at the top left of a worksheet.\n\n        Args:\n            row: The cell row (zero indexed).\n            col: The cell column (zero indexed).\n\n        Returns:\n            0:  Nothing.\n        '
        if row == 0 and col == 0:
            return
        self.top_left_cell = xl_rowcol_to_cell(row, col)

    def outline_settings(self, visible=1, symbols_below=1, symbols_right=1, auto_style=0):
        if False:
            while True:
                i = 10
        '\n        Control outline settings.\n\n        Args:\n            visible:       Outlines are visible. Optional, defaults to True.\n            symbols_below: Show row outline symbols below the outline bar.\n                           Optional, defaults to True.\n            symbols_right: Show column outline symbols to the right of the\n                           outline bar. Optional, defaults to True.\n            auto_style:    Use Automatic style. Optional, defaults to False.\n\n        Returns:\n            0:  Nothing.\n        '
        self.outline_on = visible
        self.outline_below = symbols_below
        self.outline_right = symbols_right
        self.outline_style = auto_style
        self.outline_changed = True

    @convert_cell_args
    def freeze_panes(self, row, col, top_row=None, left_col=None, pane_type=0):
        if False:
            print('Hello World!')
        '\n        Create worksheet panes and mark them as frozen.\n\n        Args:\n            row:      The cell row (zero indexed).\n            col:      The cell column (zero indexed).\n            top_row:  Topmost visible row in scrolling region of pane.\n            left_col: Leftmost visible row in scrolling region of pane.\n\n        Returns:\n            0:  Nothing.\n\n        '
        if top_row is None:
            top_row = row
        if left_col is None:
            left_col = col
        self.panes = [row, col, top_row, left_col, pane_type]

    @convert_cell_args
    def split_panes(self, x, y, top_row=None, left_col=None):
        if False:
            while True:
                i = 10
        '\n        Create worksheet panes and mark them as split.\n\n        Args:\n            x:        The position for the vertical split.\n            y:        The position for the horizontal split.\n            top_row:  Topmost visible row in scrolling region of pane.\n            left_col: Leftmost visible row in scrolling region of pane.\n\n        Returns:\n            0:  Nothing.\n\n        '
        self.freeze_panes(x, y, top_row, left_col, 2)

    def set_zoom(self, zoom=100):
        if False:
            while True:
                i = 10
        '\n        Set the worksheet zoom factor.\n\n        Args:\n            zoom: Scale factor: 10 <= zoom <= 400.\n\n        Returns:\n            Nothing.\n\n        '
        if zoom < 10 or zoom > 400:
            warn('Zoom factor %d outside range: 10 <= zoom <= 400' % zoom)
            zoom = 100
        self.zoom = int(zoom)

    def right_to_left(self):
        if False:
            return 10
        '\n        Display the worksheet right to left for some versions of Excel.\n\n        Args:\n            None.\n\n        Returns:\n            Nothing.\n\n        '
        self.is_right_to_left = 1

    def hide_zero(self):
        if False:
            i = 10
            return i + 15
        '\n        Hide zero values in worksheet cells.\n\n        Args:\n            None.\n\n        Returns:\n            Nothing.\n\n        '
        self.show_zeros = 0

    def set_tab_color(self, color):
        if False:
            i = 10
            return i + 15
        '\n        Set the color of the worksheet tab.\n\n        Args:\n            color: A #RGB color index.\n\n        Returns:\n            Nothing.\n\n        '
        self.tab_color = xl_color(color)

    def protect(self, password='', options=None):
        if False:
            i = 10
            return i + 15
        '\n        Set the password and protection options of the worksheet.\n\n        Args:\n            password: An optional password string.\n            options:  A dictionary of worksheet objects to protect.\n\n        Returns:\n            Nothing.\n\n        '
        if password != '':
            password = self._encode_password(password)
        if not options:
            options = {}
        defaults = {'sheet': True, 'content': False, 'objects': False, 'scenarios': False, 'format_cells': False, 'format_columns': False, 'format_rows': False, 'insert_columns': False, 'insert_rows': False, 'insert_hyperlinks': False, 'delete_columns': False, 'delete_rows': False, 'select_locked_cells': True, 'sort': False, 'autofilter': False, 'pivot_tables': False, 'select_unlocked_cells': True}
        for key in options.keys():
            if key in defaults:
                defaults[key] = options[key]
            else:
                warn("Unknown protection object: '%s'" % key)
        defaults['password'] = password
        self.protect_options = defaults

    def unprotect_range(self, cell_range, range_name=None, password=None):
        if False:
            while True:
                i = 10
        '\n        Unprotect ranges within a protected worksheet.\n\n        Args:\n            cell_range: The cell or cell range to unprotect.\n            range_name: An optional name for the range.\n            password:   An optional password string. (undocumented)\n\n        Returns:\n            Nothing.\n\n        '
        if cell_range is None:
            warn('Cell range must be specified in unprotect_range()')
            return -1
        cell_range = cell_range.lstrip('=')
        cell_range = cell_range.replace('$', '')
        self.num_protected_ranges += 1
        if range_name is None:
            range_name = 'Range' + str(self.num_protected_ranges)
        if password:
            password = self._encode_password(password)
        self.protected_ranges.append((cell_range, range_name, password))

    @convert_cell_args
    def insert_button(self, row, col, options=None):
        if False:
            while True:
                i = 10
        '\n        Insert a button form object into the worksheet.\n\n        Args:\n            row:     The cell row (zero indexed).\n            col:     The cell column (zero indexed).\n            options: Button formatting options.\n\n        Returns:\n            0:  Success.\n            -1: Row or column is out of worksheet bounds.\n\n        '
        if self._check_dimensions(row, col, True, True):
            warn('Cannot insert button at (%d, %d).' % (row, col))
            return -1
        if options is None:
            options = {}
        button = self._button_params(row, col, options)
        self.buttons_list.append(button)
        self.has_vml = 1
        return 0

    def set_landscape(self):
        if False:
            while True:
                i = 10
        '\n        Set the page orientation as landscape.\n\n        Args:\n            None.\n\n        Returns:\n            Nothing.\n\n        '
        self.orientation = 0
        self.page_setup_changed = True

    def set_portrait(self):
        if False:
            i = 10
            return i + 15
        '\n        Set the page orientation as portrait.\n\n        Args:\n            None.\n\n        Returns:\n            Nothing.\n\n        '
        self.orientation = 1
        self.page_setup_changed = True

    def set_page_view(self, view=1):
        if False:
            print('Hello World!')
        '\n        Set the page view mode.\n\n        Args:\n            0: Normal view mode\n            1: Page view mode (the default)\n            2: Page break view mode\n\n        Returns:\n            Nothing.\n\n        '
        self.page_view = view

    def set_pagebreak_view(self, view=1):
        if False:
            i = 10
            return i + 15
        '\n        Set the page view mode.\n\n        Args:\n            None.\n\n        Returns:\n            Nothing.\n\n        '
        self.page_view = 2

    def set_paper(self, paper_size):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the paper type. US Letter = 1, A4 = 9.\n\n        Args:\n            paper_size: Paper index.\n\n        Returns:\n            Nothing.\n\n        '
        if paper_size:
            self.paper_size = paper_size
            self.page_setup_changed = True

    def center_horizontally(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Center the page horizontally.\n\n        Args:\n            None.\n\n        Returns:\n            Nothing.\n\n        '
        self.print_options_changed = True
        self.hcenter = 1

    def center_vertically(self):
        if False:
            while True:
                i = 10
        '\n        Center the page vertically.\n\n        Args:\n            None.\n\n        Returns:\n            Nothing.\n\n        '
        self.print_options_changed = True
        self.vcenter = 1

    def set_margins(self, left=0.7, right=0.7, top=0.75, bottom=0.75):
        if False:
            print('Hello World!')
        '\n        Set all the page margins in inches.\n\n        Args:\n            left:   Left margin.\n            right:  Right margin.\n            top:    Top margin.\n            bottom: Bottom margin.\n\n        Returns:\n            Nothing.\n\n        '
        self.margin_left = left
        self.margin_right = right
        self.margin_top = top
        self.margin_bottom = bottom

    def set_header(self, header='', options=None, margin=None):
        if False:
            i = 10
            return i + 15
        '\n        Set the page header caption and optional margin.\n\n        Args:\n            header:  Header string.\n            margin:  Header margin.\n            options: Header options, mainly for images.\n\n        Returns:\n            Nothing.\n\n        '
        header_orig = header
        header = header.replace('&[Picture]', '&G')
        if len(header) > 255:
            warn("Header string cannot be longer than Excel's limit of 255 characters")
            return
        if options is not None:
            if not isinstance(options, dict):
                options = {'margin': options}
        else:
            options = {}
        options = options.copy()
        if margin is not None:
            options['margin'] = margin
        self.header_images = []
        if options.get('image_left'):
            self.header_images.append([options.get('image_left'), options.get('image_data_left'), 'LH'])
        if options.get('image_center'):
            self.header_images.append([options.get('image_center'), options.get('image_data_center'), 'CH'])
        if options.get('image_right'):
            self.header_images.append([options.get('image_right'), options.get('image_data_right'), 'RH'])
        placeholder_count = header.count('&G')
        image_count = len(self.header_images)
        if placeholder_count != image_count:
            warn("Number of header images (%s) doesn't match placeholder count (%s) in string: %s" % (image_count, placeholder_count, header_orig))
            self.header_images = []
            return
        if 'align_with_margins' in options:
            self.header_footer_aligns = options['align_with_margins']
        if 'scale_with_doc' in options:
            self.header_footer_scales = options['scale_with_doc']
        self.header = header
        self.margin_header = options.get('margin', 0.3)
        self.header_footer_changed = True
        if image_count:
            self.has_header_vml = True

    def set_footer(self, footer='', options=None, margin=None):
        if False:
            while True:
                i = 10
        '\n        Set the page footer caption and optional margin.\n\n        Args:\n            footer:  Footer string.\n            margin:  Footer margin.\n            options: Footer options, mainly for images.\n\n        Returns:\n            Nothing.\n\n        '
        footer_orig = footer
        footer = footer.replace('&[Picture]', '&G')
        if len(footer) > 255:
            warn("Footer string cannot be longer than Excel's limit of 255 characters")
            return
        if options is not None:
            if not isinstance(options, dict):
                options = {'margin': options}
        else:
            options = {}
        options = options.copy()
        if margin is not None:
            options['margin'] = margin
        self.footer_images = []
        if options.get('image_left'):
            self.footer_images.append([options.get('image_left'), options.get('image_data_left'), 'LF'])
        if options.get('image_center'):
            self.footer_images.append([options.get('image_center'), options.get('image_data_center'), 'CF'])
        if options.get('image_right'):
            self.footer_images.append([options.get('image_right'), options.get('image_data_right'), 'RF'])
        placeholder_count = footer.count('&G')
        image_count = len(self.footer_images)
        if placeholder_count != image_count:
            warn("Number of footer images (%s) doesn't match placeholder count (%s) in string: %s" % (image_count, placeholder_count, footer_orig))
            self.footer_images = []
            return
        if 'align_with_margins' in options:
            self.header_footer_aligns = options['align_with_margins']
        if 'scale_with_doc' in options:
            self.header_footer_scales = options['scale_with_doc']
        self.footer = footer
        self.margin_footer = options.get('margin', 0.3)
        self.header_footer_changed = True
        if image_count:
            self.has_header_vml = True

    def repeat_rows(self, first_row, last_row=None):
        if False:
            i = 10
            return i + 15
        '\n        Set the rows to repeat at the top of each printed page.\n\n        Args:\n            first_row: Start row for range.\n            last_row: End row for range.\n\n        Returns:\n            Nothing.\n\n        '
        if last_row is None:
            last_row = first_row
        first_row += 1
        last_row += 1
        area = '$%d:$%d' % (first_row, last_row)
        sheetname = quote_sheetname(self.name)
        self.repeat_row_range = sheetname + '!' + area

    @convert_column_args
    def repeat_columns(self, first_col, last_col=None):
        if False:
            print('Hello World!')
        '\n        Set the columns to repeat at the left hand side of each printed page.\n\n        Args:\n            first_col: Start column for range.\n            last_col: End column for range.\n\n        Returns:\n            Nothing.\n\n        '
        if last_col is None:
            last_col = first_col
        first_col = xl_col_to_name(first_col, 1)
        last_col = xl_col_to_name(last_col, 1)
        area = first_col + ':' + last_col
        sheetname = quote_sheetname(self.name)
        self.repeat_col_range = sheetname + '!' + area

    def hide_gridlines(self, option=1):
        if False:
            return 10
        "\n        Set the option to hide gridlines on the screen and the printed page.\n\n        Args:\n            option:    0 : Don't hide gridlines\n                       1 : Hide printed gridlines only\n                       2 : Hide screen and printed gridlines\n\n        Returns:\n            Nothing.\n\n        "
        if option == 0:
            self.print_gridlines = 1
            self.screen_gridlines = 1
            self.print_options_changed = True
        elif option == 1:
            self.print_gridlines = 0
            self.screen_gridlines = 1
        else:
            self.print_gridlines = 0
            self.screen_gridlines = 0

    def print_row_col_headers(self):
        if False:
            i = 10
            return i + 15
        '\n        Set the option to print the row and column headers on the printed page.\n\n        Args:\n            None.\n\n        Returns:\n            Nothing.\n\n        '
        self.print_headers = True
        self.print_options_changed = True

    def hide_row_col_headers(self):
        if False:
            print('Hello World!')
        '\n        Set the option to hide the row and column headers on the worksheet.\n\n        Args:\n            None.\n\n        Returns:\n            Nothing.\n\n        '
        self.row_col_headers = True

    @convert_range_args
    def print_area(self, first_row, first_col, last_row, last_col):
        if False:
            i = 10
            return i + 15
        '\n        Set the print area in the current worksheet.\n\n        Args:\n            first_row:    The first row of the cell range. (zero indexed).\n            first_col:    The first column of the cell range.\n            last_row:     The last row of the cell range. (zero indexed).\n            last_col:     The last column of the cell range.\n\n        Returns:\n            0:  Success.\n            -1: Row or column is out of worksheet bounds.\n\n        '
        if first_row == 0 and first_col == 0 and (last_row == self.xls_rowmax - 1) and (last_col == self.xls_colmax - 1):
            return
        area = self._convert_name_area(first_row, first_col, last_row, last_col)
        self.print_area_range = area
        return 0

    def print_across(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the order in which pages are printed.\n\n        Args:\n            None.\n\n        Returns:\n            Nothing.\n\n        '
        self.page_order = 1
        self.page_setup_changed = True

    def fit_to_pages(self, width, height):
        if False:
            while True:
                i = 10
        '\n        Fit the printed area to a specific number of pages both vertically and\n        horizontally.\n\n        Args:\n            width:  Number of pages horizontally.\n            height: Number of pages vertically.\n\n        Returns:\n            Nothing.\n\n        '
        self.fit_page = 1
        self.fit_width = width
        self.fit_height = height
        self.page_setup_changed = True

    def set_start_page(self, start_page):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the start page number when printing.\n\n        Args:\n            start_page: Start page number.\n\n        Returns:\n            Nothing.\n\n        '
        self.page_start = start_page

    def set_print_scale(self, scale):
        if False:
            while True:
                i = 10
        '\n        Set the scale factor for the printed page.\n\n        Args:\n            scale: Print scale. 10 <= scale <= 400.\n\n        Returns:\n            Nothing.\n\n        '
        if scale < 10 or scale > 400:
            warn("Print scale '%d' outside range: 10 <= scale <= 400" % scale)
            return
        self.fit_page = 0
        self.print_scale = int(scale)
        self.page_setup_changed = True

    def print_black_and_white(self):
        if False:
            i = 10
            return i + 15
        '\n        Set the option to print the worksheet in black and white.\n\n        Args:\n            None.\n\n        Returns:\n            Nothing.\n\n        '
        self.black_white = True
        self.page_setup_changed = True

    def set_h_pagebreaks(self, breaks):
        if False:
            print('Hello World!')
        '\n        Set the horizontal page breaks on a worksheet.\n\n        Args:\n            breaks: List of rows where the page breaks should be added.\n\n        Returns:\n            Nothing.\n\n        '
        self.hbreaks = breaks

    def set_v_pagebreaks(self, breaks):
        if False:
            i = 10
            return i + 15
        '\n        Set the horizontal page breaks on a worksheet.\n\n        Args:\n            breaks: List of columns where the page breaks should be added.\n\n        Returns:\n            Nothing.\n\n        '
        self.vbreaks = breaks

    def set_vba_name(self, name=None):
        if False:
            print('Hello World!')
        '\n        Set the VBA name for the worksheet. By default this is the\n        same as the sheet name: i.e., Sheet1 etc.\n\n        Args:\n            name: The VBA name for the worksheet.\n\n        Returns:\n            Nothing.\n\n        '
        if name is not None:
            self.vba_codename = name
        else:
            self.vba_codename = 'Sheet' + str(self.index + 1)

    def ignore_errors(self, options=None):
        if False:
            i = 10
            return i + 15
        '\n        Ignore various Excel errors/warnings in a worksheet for user defined\n        ranges.\n\n        Args:\n            options: A dict of ignore errors keys with cell range values.\n\n        Returns:\n            0: Success.\n           -1: Incorrect parameter or option.\n\n        '
        if options is None:
            return -1
        else:
            options = options.copy()
        valid_parameters = {'number_stored_as_text', 'eval_error', 'formula_differs', 'formula_range', 'formula_unlocked', 'empty_cell_reference', 'list_data_validation', 'calculated_column', 'two_digit_text_year'}
        for param_key in options.keys():
            if param_key not in valid_parameters:
                warn("Unknown parameter '%s' in ignore_errors()" % param_key)
                return -1
        self.ignored_errors = options
        return 0

    def _initialize(self, init_data):
        if False:
            while True:
                i = 10
        self.name = init_data['name']
        self.index = init_data['index']
        self.str_table = init_data['str_table']
        self.worksheet_meta = init_data['worksheet_meta']
        self.constant_memory = init_data['constant_memory']
        self.tmpdir = init_data['tmpdir']
        self.date_1904 = init_data['date_1904']
        self.strings_to_numbers = init_data['strings_to_numbers']
        self.strings_to_formulas = init_data['strings_to_formulas']
        self.strings_to_urls = init_data['strings_to_urls']
        self.nan_inf_to_errors = init_data['nan_inf_to_errors']
        self.default_date_format = init_data['default_date_format']
        self.default_url_format = init_data['default_url_format']
        self.excel2003_style = init_data['excel2003_style']
        self.remove_timezone = init_data['remove_timezone']
        self.max_url_length = init_data['max_url_length']
        self.use_future_functions = init_data['use_future_functions']
        if self.excel2003_style:
            self.original_row_height = 12.75
            self.default_row_height = 12.75
            self.default_row_pixels = 17
            self.margin_left = 0.75
            self.margin_right = 0.75
            self.margin_top = 1
            self.margin_bottom = 1
            self.margin_header = 0.5
            self.margin_footer = 0.5
            self.header_footer_aligns = False
        if self.constant_memory:
            (fd, filename) = tempfile.mkstemp(dir=self.tmpdir)
            os.close(fd)
            self.row_data_filename = filename
            self.row_data_fh = open(filename, mode='w+', encoding='utf-8')
            self.fh = self.row_data_fh

    def _assemble_xml_file(self):
        if False:
            while True:
                i = 10
        self._xml_declaration()
        self._write_worksheet()
        self._write_sheet_pr()
        self._write_dimension()
        self._write_sheet_views()
        self._write_sheet_format_pr()
        self._write_cols()
        if not self.constant_memory:
            self._write_sheet_data()
        else:
            self._write_optimized_sheet_data()
        self._write_sheet_protection()
        self._write_protected_ranges()
        if self.excel2003_style:
            self._write_phonetic_pr()
        self._write_auto_filter()
        self._write_merge_cells()
        self._write_conditional_formats()
        self._write_data_validations()
        self._write_hyperlinks()
        self._write_print_options()
        self._write_page_margins()
        self._write_page_setup()
        self._write_header_footer()
        self._write_row_breaks()
        self._write_col_breaks()
        self._write_ignored_errors()
        self._write_drawings()
        self._write_legacy_drawing()
        self._write_legacy_drawing_hf()
        self._write_picture()
        self._write_table_parts()
        self._write_ext_list()
        self._xml_end_tag('worksheet')
        self._xml_close()

    def _check_dimensions(self, row, col, ignore_row=False, ignore_col=False):
        if False:
            return 10
        if row < 0 or col < 0:
            return -1
        if row >= self.xls_rowmax or col >= self.xls_colmax:
            return -1
        if not ignore_row and (not ignore_col) and self.constant_memory:
            if row < self.previous_row:
                return -2
        if not ignore_row:
            if self.dim_rowmin is None or row < self.dim_rowmin:
                self.dim_rowmin = row
            if self.dim_rowmax is None or row > self.dim_rowmax:
                self.dim_rowmax = row
        if not ignore_col:
            if self.dim_colmin is None or col < self.dim_colmin:
                self.dim_colmin = col
            if self.dim_colmax is None or col > self.dim_colmax:
                self.dim_colmax = col
        return 0

    def _convert_date_time(self, dt_obj):
        if False:
            while True:
                i = 10
        return datetime_to_excel_datetime(dt_obj, self.date_1904, self.remove_timezone)

    def _convert_name_area(self, row_num_1, col_num_1, row_num_2, col_num_2):
        if False:
            print('Hello World!')
        range1 = ''
        range2 = ''
        area = ''
        row_col_only = 0
        col_char_1 = xl_col_to_name(col_num_1, 1)
        col_char_2 = xl_col_to_name(col_num_2, 1)
        row_char_1 = '$' + str(row_num_1 + 1)
        row_char_2 = '$' + str(row_num_2 + 1)
        if row_num_1 == 0 and row_num_2 == self.xls_rowmax - 1:
            range1 = col_char_1
            range2 = col_char_2
            row_col_only = 1
        elif col_num_1 == 0 and col_num_2 == self.xls_colmax - 1:
            range1 = row_char_1
            range2 = row_char_2
            row_col_only = 1
        else:
            range1 = col_char_1 + row_char_1
            range2 = col_char_2 + row_char_2
        if range1 == range2 and (not row_col_only):
            area = range1
        else:
            area = range1 + ':' + range2
        sheetname = quote_sheetname(self.name)
        area = sheetname + '!' + area
        return area

    def _sort_pagebreaks(self, breaks):
        if False:
            while True:
                i = 10
        if not breaks:
            return
        breaks_set = set(breaks)
        if 0 in breaks_set:
            breaks_set.remove(0)
        breaks_list = list(breaks_set)
        breaks_list.sort()
        max_num_breaks = 1023
        if len(breaks_list) > max_num_breaks:
            breaks_list = breaks_list[:max_num_breaks]
        return breaks_list

    def _extract_filter_tokens(self, expression):
        if False:
            return 10
        if not expression:
            return []
        token_re = re.compile('"(?:[^"]|"")*"|\\S+')
        tokens = token_re.findall(expression)
        new_tokens = []
        for token in tokens:
            if token.startswith('"'):
                token = token[1:]
            if token.endswith('"'):
                token = token[:-1]
            token = token.replace('""', '"')
            new_tokens.append(token)
        return new_tokens

    def _parse_filter_expression(self, expression, tokens):
        if False:
            for i in range(10):
                print('nop')
        if len(tokens) == 7:
            conditional = tokens[3]
            if re.match('(and|&&)', conditional):
                conditional = 0
            elif re.match('(or|\\|\\|)', conditional):
                conditional = 1
            else:
                warn("Token '%s' is not a valid conditional in filter expression '%s'" % (conditional, expression))
            expression_1 = self._parse_filter_tokens(expression, tokens[0:3])
            expression_2 = self._parse_filter_tokens(expression, tokens[4:7])
            return expression_1 + [conditional] + expression_2
        else:
            return self._parse_filter_tokens(expression, tokens)

    def _parse_filter_tokens(self, expression, tokens):
        if False:
            i = 10
            return i + 15
        operators = {'==': 2, '=': 2, '=~': 2, 'eq': 2, '!=': 5, '!~': 5, 'ne': 5, '<>': 5, '<': 1, '<=': 3, '>': 4, '>=': 6}
        operator = operators.get(tokens[1], None)
        token = tokens[2]
        if re.match('top|bottom', tokens[0].lower()):
            value = int(tokens[1])
            if value < 1 or value > 500:
                warn("The value '%d' in expression '%s' must be in the range 1 to 500" % (value, expression))
            token = token.lower()
            if token != 'items' and token != '%':
                warn("The type '%s' in expression '%s' must be either 'items' or '%%'" % (token, expression))
            if tokens[0].lower() == 'top':
                operator = 30
            else:
                operator = 32
            if tokens[2] == '%':
                operator += 1
            token = str(value)
        if not operator and tokens[0]:
            warn("Token '%s' is not a valid operator in filter expression '%s'" % (token[0], expression))
        if re.match('blanks|nonblanks', token.lower()):
            if operator != 2 and operator != 5:
                warn("The operator '%s' in expression '%s' is not valid in relation to Blanks/NonBlanks'" % (tokens[1], expression))
            token = token.lower()
            if token == 'blanks':
                if operator == 5:
                    token = ' '
            elif operator == 5:
                operator = 2
                token = 'blanks'
            else:
                operator = 5
                token = ' '
        if operator == 2 and re.search('[*?]', token):
            operator = 22
        return [operator, token]

    def _encode_password(self, password):
        if False:
            for i in range(10):
                print('nop')
        hash = 0
        for char in password[::-1]:
            hash = hash >> 14 & 1 | hash << 1 & 32767
            hash ^= ord(char)
        hash = hash >> 14 & 1 | hash << 1 & 32767
        hash ^= len(password)
        hash ^= 52811
        return '%X' % hash

    def _prepare_image(self, index, image_id, drawing_id, width, height, name, image_type, x_dpi, y_dpi, digest):
        if False:
            for i in range(10):
                print('nop')
        drawing_type = 2
        (row, col, _, x_offset, y_offset, x_scale, y_scale, url, tip, anchor, _, description, decorative) = self.images[index]
        width *= x_scale
        height *= y_scale
        width *= 96.0 / x_dpi
        height *= 96.0 / y_dpi
        dimensions = self._position_object_emus(col, row, x_offset, y_offset, width, height, anchor)
        width = int(0.5 + width * 9525)
        height = int(0.5 + height * 9525)
        if not self.drawing:
            drawing = Drawing()
            drawing.embedded = 1
            self.drawing = drawing
            self.external_drawing_links.append(['/drawing', '../drawings/drawing' + str(drawing_id) + '.xml', None])
        else:
            drawing = self.drawing
        drawing_object = drawing._add_drawing_object()
        drawing_object['type'] = drawing_type
        drawing_object['dimensions'] = dimensions
        drawing_object['width'] = width
        drawing_object['height'] = height
        drawing_object['description'] = name
        drawing_object['shape'] = None
        drawing_object['anchor'] = anchor
        drawing_object['rel_index'] = 0
        drawing_object['url_rel_index'] = 0
        drawing_object['tip'] = tip
        drawing_object['decorative'] = decorative
        if description is not None:
            drawing_object['description'] = description
        if url:
            target = None
            rel_type = '/hyperlink'
            target_mode = 'External'
            if re.match('(ftp|http)s?://', url):
                target = self._escape_url(url)
            if re.match('^mailto:', url):
                target = self._escape_url(url)
            if re.match('external:', url):
                target = url.replace('external:', '')
                target = self._escape_url(target)
                target = target.replace('#', '%23')
                if re.match('\\w:', target) or re.match('\\\\', target):
                    target = 'file:///' + target
                else:
                    target = target.replace('\\', '/')
            if re.match('internal:', url):
                target = url.replace('internal:', '#')
                target_mode = None
            if target is not None:
                if len(target) > self.max_url_length:
                    warn("Ignoring URL '%s' with link and/or anchor > %d characters since it exceeds Excel's limit for URLS" % (url, self.max_url_length))
                else:
                    if not self.drawing_rels.get(url):
                        self.drawing_links.append([rel_type, target, target_mode])
                    drawing_object['url_rel_index'] = self._get_drawing_rel_index(url)
        if not self.drawing_rels.get(digest):
            self.drawing_links.append(['/image', '../media/image' + str(image_id) + '.' + image_type])
        drawing_object['rel_index'] = self._get_drawing_rel_index(digest)

    def _prepare_shape(self, index, drawing_id):
        if False:
            while True:
                i = 10
        drawing_type = 3
        (row, col, x_offset, y_offset, x_scale, y_scale, text, anchor, options, description, decorative) = self.shapes[index]
        width = options.get('width', self.default_col_pixels * 3)
        height = options.get('height', self.default_row_pixels * 6)
        width *= x_scale
        height *= y_scale
        dimensions = self._position_object_emus(col, row, x_offset, y_offset, width, height, anchor)
        width = int(0.5 + width * 9525)
        height = int(0.5 + height * 9525)
        if not self.drawing:
            drawing = Drawing()
            drawing.embedded = 1
            self.drawing = drawing
            self.external_drawing_links.append(['/drawing', '../drawings/drawing' + str(drawing_id) + '.xml', None])
        else:
            drawing = self.drawing
        shape = Shape('rect', 'TextBox', options)
        shape.text = text
        drawing_object = drawing._add_drawing_object()
        drawing_object['type'] = drawing_type
        drawing_object['dimensions'] = dimensions
        drawing_object['width'] = width
        drawing_object['height'] = height
        drawing_object['description'] = description
        drawing_object['shape'] = shape
        drawing_object['anchor'] = anchor
        drawing_object['rel_index'] = 0
        drawing_object['url_rel_index'] = 0
        drawing_object['tip'] = options.get('tip')
        drawing_object['decorative'] = decorative
        url = options.get('url', None)
        if url:
            target = None
            rel_type = '/hyperlink'
            target_mode = 'External'
            if re.match('(ftp|http)s?://', url):
                target = self._escape_url(url)
            if re.match('^mailto:', url):
                target = self._escape_url(url)
            if re.match('external:', url):
                target = url.replace('external:', 'file:///')
                target = self._escape_url(target)
                target = target.replace('#', '%23')
            if re.match('internal:', url):
                target = url.replace('internal:', '#')
                target_mode = None
            if target is not None:
                if len(target) > self.max_url_length:
                    warn("Ignoring URL '%s' with link and/or anchor > %d characters since it exceeds Excel's limit for URLS" % (url, self.max_url_length))
                else:
                    if not self.drawing_rels.get(url):
                        self.drawing_links.append([rel_type, target, target_mode])
                    drawing_object['url_rel_index'] = self._get_drawing_rel_index(url)

    def _prepare_header_image(self, image_id, width, height, name, image_type, position, x_dpi, y_dpi, digest):
        if False:
            i = 10
            return i + 15
        name = re.sub('\\..*$', '', name)
        if not self.vml_drawing_rels.get(digest):
            self.vml_drawing_links.append(['/image', '../media/image' + str(image_id) + '.' + image_type])
        ref_id = self._get_vml_drawing_rel_index(digest)
        self.header_images_list.append([width, height, name, position, x_dpi, y_dpi, ref_id])

    def _prepare_background(self, image_id, image_type):
        if False:
            i = 10
            return i + 15
        self.external_background_links.append(['/image', '../media/image' + str(image_id) + '.' + image_type])

    def _prepare_chart(self, index, chart_id, drawing_id):
        if False:
            print('Hello World!')
        drawing_type = 1
        (row, col, chart, x_offset, y_offset, x_scale, y_scale, anchor, description, decorative) = self.charts[index]
        chart.id = chart_id - 1
        width = int(0.5 + chart.width * x_scale)
        height = int(0.5 + chart.height * y_scale)
        dimensions = self._position_object_emus(col, row, x_offset, y_offset, width, height, anchor)
        name = chart.chart_name
        if not self.drawing:
            drawing = Drawing()
            drawing.embedded = 1
            self.drawing = drawing
            self.external_drawing_links.append(['/drawing', '../drawings/drawing' + str(drawing_id) + '.xml'])
        else:
            drawing = self.drawing
        drawing_object = drawing._add_drawing_object()
        drawing_object['type'] = drawing_type
        drawing_object['dimensions'] = dimensions
        drawing_object['width'] = width
        drawing_object['height'] = height
        drawing_object['name'] = name
        drawing_object['shape'] = None
        drawing_object['anchor'] = anchor
        drawing_object['rel_index'] = self._get_drawing_rel_index()
        drawing_object['url_rel_index'] = 0
        drawing_object['tip'] = None
        drawing_object['description'] = description
        drawing_object['decorative'] = decorative
        self.drawing_links.append(['/chart', '../charts/chart' + str(chart_id) + '.xml'])

    def _position_object_emus(self, col_start, row_start, x1, y1, width, height, anchor):
        if False:
            i = 10
            return i + 15
        (col_start, row_start, x1, y1, col_end, row_end, x2, y2, x_abs, y_abs) = self._position_object_pixels(col_start, row_start, x1, y1, width, height, anchor)
        x1 = int(0.5 + 9525 * x1)
        y1 = int(0.5 + 9525 * y1)
        x2 = int(0.5 + 9525 * x2)
        y2 = int(0.5 + 9525 * y2)
        x_abs = int(0.5 + 9525 * x_abs)
        y_abs = int(0.5 + 9525 * y_abs)
        return (col_start, row_start, x1, y1, col_end, row_end, x2, y2, x_abs, y_abs)

    def _position_object_pixels(self, col_start, row_start, x1, y1, width, height, anchor):
        if False:
            print('Hello World!')
        x_abs = 0
        y_abs = 0
        while x1 < 0 and col_start > 0:
            x1 += self._size_col(col_start - 1)
            col_start -= 1
        while y1 < 0 and row_start > 0:
            y1 += self._size_row(row_start - 1)
            row_start -= 1
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if self.col_size_changed:
            for col_id in range(col_start):
                x_abs += self._size_col(col_id)
        else:
            x_abs += self.default_col_pixels * col_start
        x_abs += x1
        if self.row_size_changed:
            for row_id in range(row_start):
                y_abs += self._size_row(row_id)
        else:
            y_abs += self.default_row_pixels * row_start
        y_abs += y1
        while x1 >= self._size_col(col_start, anchor):
            x1 -= self._size_col(col_start)
            col_start += 1
        while y1 >= self._size_row(row_start, anchor):
            y1 -= self._size_row(row_start)
            row_start += 1
        col_end = col_start
        row_end = row_start
        if self._size_col(col_start, anchor) > 0:
            width = width + x1
        if self._size_row(row_start, anchor) > 0:
            height = height + y1
        while width >= self._size_col(col_end, anchor):
            width -= self._size_col(col_end, anchor)
            col_end += 1
        while height >= self._size_row(row_end, anchor):
            height -= self._size_row(row_end, anchor)
            row_end += 1
        x2 = width
        y2 = height
        return [col_start, row_start, x1, y1, col_end, row_end, x2, y2, x_abs, y_abs]

    def _size_col(self, col, anchor=0):
        if False:
            i = 10
            return i + 15
        max_digit_width = 7
        padding = 5
        pixels = 0
        if col in self.col_info:
            width = self.col_info[col][0]
            hidden = self.col_info[col][2]
            if width is None:
                width = self.default_col_width
            if hidden and anchor != 4:
                pixels = 0
            elif width < 1:
                pixels = int(width * (max_digit_width + padding) + 0.5)
            else:
                pixels = int(width * max_digit_width + 0.5) + padding
        else:
            pixels = self.default_col_pixels
        return pixels

    def _size_row(self, row, anchor=0):
        if False:
            while True:
                i = 10
        pixels = 0
        if row in self.row_sizes:
            height = self.row_sizes[row][0]
            hidden = self.row_sizes[row][1]
            if hidden and anchor != 4:
                pixels = 0
            else:
                pixels = int(4.0 / 3.0 * height)
        else:
            pixels = int(4.0 / 3.0 * self.default_row_height)
        return pixels

    def _pixels_to_width(self, pixels):
        if False:
            print('Hello World!')
        max_digit_width = 7.0
        padding = 5.0
        if pixels <= 12:
            width = pixels / (max_digit_width + padding)
        else:
            width = (pixels - padding) / max_digit_width
        return width

    def _pixels_to_height(self, pixels):
        if False:
            i = 10
            return i + 15
        return 0.75 * pixels

    def _comment_params(self, row, col, string, options):
        if False:
            print('Hello World!')
        default_width = 128
        default_height = 74
        anchor = 0
        params = {'author': None, 'color': '#ffffe1', 'start_cell': None, 'start_col': None, 'start_row': None, 'visible': None, 'width': default_width, 'height': default_height, 'x_offset': None, 'x_scale': 1, 'y_offset': None, 'y_scale': 1, 'font_name': 'Tahoma', 'font_size': 8, 'font_family': 2}
        for key in options.keys():
            params[key] = options[key]
        if not params['width']:
            params['width'] = default_width
        if not params['height']:
            params['height'] = default_height
        params['color'] = xl_color(params['color']).lower()
        params['color'] = params['color'].replace('ff', '#', 1)
        if params['start_cell'] is not None:
            (start_row, start_col) = xl_cell_to_rowcol(params['start_cell'])
            params['start_row'] = start_row
            params['start_col'] = start_col
        row_max = self.xls_rowmax
        col_max = self.xls_colmax
        if params['start_row'] is None:
            if row == 0:
                params['start_row'] = 0
            elif row == row_max - 3:
                params['start_row'] = row_max - 7
            elif row == row_max - 2:
                params['start_row'] = row_max - 6
            elif row == row_max - 1:
                params['start_row'] = row_max - 5
            else:
                params['start_row'] = row - 1
        if params['y_offset'] is None:
            if row == 0:
                params['y_offset'] = 2
            elif row == row_max - 3:
                params['y_offset'] = 16
            elif row == row_max - 2:
                params['y_offset'] = 16
            elif row == row_max - 1:
                params['y_offset'] = 14
            else:
                params['y_offset'] = 10
        if params['start_col'] is None:
            if col == col_max - 3:
                params['start_col'] = col_max - 6
            elif col == col_max - 2:
                params['start_col'] = col_max - 5
            elif col == col_max - 1:
                params['start_col'] = col_max - 4
            else:
                params['start_col'] = col + 1
        if params['x_offset'] is None:
            if col == col_max - 3:
                params['x_offset'] = 49
            elif col == col_max - 2:
                params['x_offset'] = 49
            elif col == col_max - 1:
                params['x_offset'] = 49
            else:
                params['x_offset'] = 15
        if params['x_scale']:
            params['width'] = params['width'] * params['x_scale']
        if params['y_scale']:
            params['height'] = params['height'] * params['y_scale']
        params['width'] = int(0.5 + params['width'])
        params['height'] = int(0.5 + params['height'])
        vertices = self._position_object_pixels(params['start_col'], params['start_row'], params['x_offset'], params['y_offset'], params['width'], params['height'], anchor)
        vertices.append(params['width'])
        vertices.append(params['height'])
        return [row, col, string, params['author'], params['visible'], params['color'], params['font_name'], params['font_size'], params['font_family']] + [vertices]

    def _button_params(self, row, col, options):
        if False:
            return 10
        default_height = self.default_row_pixels
        default_width = self.default_col_pixels
        anchor = 0
        button_number = 1 + len(self.buttons_list)
        button = {'row': row, 'col': col, 'font': {}}
        params = {}
        for key in options.keys():
            params[key] = options[key]
        caption = params.get('caption')
        if caption is None:
            caption = 'Button %d' % button_number
        button['font']['caption'] = caption
        if params.get('macro'):
            button['macro'] = '[0]!' + params['macro']
        else:
            button['macro'] = '[0]!Button%d_Click' % button_number
        button['description'] = params.get('description')
        params['width'] = params.get('width', default_width)
        params['height'] = params.get('height', default_height)
        params['x_offset'] = params.get('x_offset', 0)
        params['y_offset'] = params.get('y_offset', 0)
        params['width'] = params['width'] * params.get('x_scale', 1)
        params['height'] = params['height'] * params.get('y_scale', 1)
        params['width'] = int(0.5 + params['width'])
        params['height'] = int(0.5 + params['height'])
        params['start_row'] = row
        params['start_col'] = col
        vertices = self._position_object_pixels(params['start_col'], params['start_row'], params['x_offset'], params['y_offset'], params['width'], params['height'], anchor)
        vertices.append(params['width'])
        vertices.append(params['height'])
        button['vertices'] = vertices
        return button

    def _prepare_vml_objects(self, vml_data_id, vml_shape_id, vml_drawing_id, comment_id):
        if False:
            i = 10
            return i + 15
        comments = []
        row_nums = sorted(self.comments.keys())
        for row in row_nums:
            col_nums = sorted(self.comments[row].keys())
            for col in col_nums:
                user_options = self.comments[row][col]
                params = self._comment_params(*user_options)
                self.comments[row][col] = params
                if self.comments_visible:
                    if self.comments[row][col][4] is None:
                        self.comments[row][col][4] = 1
                if self.comments[row][col][3] is None:
                    self.comments[row][col][3] = self.comments_author
                comments.append(self.comments[row][col])
        self.external_vml_links.append(['/vmlDrawing', '../drawings/vmlDrawing' + str(vml_drawing_id) + '.vml'])
        if self.has_comments:
            self.comments_list = comments
            self.external_comment_links.append(['/comments', '../comments' + str(comment_id) + '.xml'])
        count = len(comments)
        start_data_id = vml_data_id
        for i in range(int(count / 1024)):
            vml_data_id = '%s,%d' % (vml_data_id, start_data_id + i + 1)
        self.vml_data_id = vml_data_id
        self.vml_shape_id = vml_shape_id
        return count

    def _prepare_header_vml_objects(self, vml_header_id, vml_drawing_id):
        if False:
            return 10
        self.vml_header_id = vml_header_id
        self.external_vml_links.append(['/vmlDrawing', '../drawings/vmlDrawing' + str(vml_drawing_id) + '.vml'])

    def _prepare_tables(self, table_id, seen):
        if False:
            print('Hello World!')
        for table in self.tables:
            table['id'] = table_id
            if table.get('name') is None:
                table['name'] = 'Table' + str(table_id)
            name = table['name'].lower()
            if name in seen:
                raise DuplicateTableName("Duplicate name '%s' used in worksheet.add_table()." % table['name'])
            else:
                seen[name] = True
            self.external_table_links.append(['/table', '../tables/table' + str(table_id) + '.xml'])
            table_id += 1

    def _table_function_to_formula(self, function, col_name):
        if False:
            for i in range(10):
                print('nop')
        formula = ''
        col_name = col_name.replace("'", "''")
        col_name = col_name.replace('#', "'#")
        col_name = col_name.replace(']', "']")
        col_name = col_name.replace('[', "'[")
        subtotals = {'average': 101, 'countNums': 102, 'count': 103, 'max': 104, 'min': 105, 'stdDev': 107, 'sum': 109, 'var': 110}
        if function in subtotals:
            func_num = subtotals[function]
            formula = 'SUBTOTAL(%s,[%s])' % (func_num, col_name)
        else:
            warn("Unsupported function '%s' in add_table()" % function)
        return formula

    def _set_spark_color(self, sparkline, options, user_color):
        if False:
            while True:
                i = 10
        if user_color not in options:
            return
        sparkline[user_color] = {'rgb': xl_color(options[user_color])}

    def _get_range_data(self, row_start, col_start, row_end, col_end):
        if False:
            return 10
        if self.constant_memory:
            return ()
        data = []
        for row_num in range(row_start, row_end + 1):
            if row_num not in self.table:
                data.append(None)
                continue
            for col_num in range(col_start, col_end + 1):
                if col_num in self.table[row_num]:
                    cell = self.table[row_num][col_num]
                    cell_type = cell.__class__.__name__
                    if cell_type in ('Number', 'Datetime'):
                        data.append('%.16g' % cell.number)
                    elif cell_type == 'String':
                        index = cell.string
                        string = self.str_table._get_shared_string(index)
                        data.append(string)
                    elif cell_type in ('Formula', 'ArrayFormula'):
                        value = cell.value
                        if value is None:
                            value = 0
                        data.append(value)
                    elif cell_type == 'Blank':
                        data.append('')
                else:
                    data.append(None)
        return data

    def _csv_join(self, *items):
        if False:
            print('Hello World!')
        items = [str(item) if not isinstance(item, str) else item for item in items]
        return ','.join(items)

    def _escape_url(self, url):
        if False:
            i = 10
            return i + 15
        if re.search('%[0-9a-fA-F]{2}', url):
            return url
        url = url.replace('%', '%25')
        url = url.replace('"', '%22')
        url = url.replace(' ', '%20')
        url = url.replace('<', '%3c')
        url = url.replace('>', '%3e')
        url = url.replace('[', '%5b')
        url = url.replace(']', '%5d')
        url = url.replace('^', '%5e')
        url = url.replace('`', '%60')
        url = url.replace('{', '%7b')
        url = url.replace('}', '%7d')
        return url

    def _get_drawing_rel_index(self, target=None):
        if False:
            while True:
                i = 10
        if target is None:
            self.drawing_rels_id += 1
            return self.drawing_rels_id
        elif self.drawing_rels.get(target):
            return self.drawing_rels[target]
        else:
            self.drawing_rels_id += 1
            self.drawing_rels[target] = self.drawing_rels_id
            return self.drawing_rels_id

    def _get_vml_drawing_rel_index(self, target=None):
        if False:
            print('Hello World!')
        if self.vml_drawing_rels.get(target):
            return self.vml_drawing_rels[target]
        else:
            self.vml_drawing_rels_id += 1
            self.vml_drawing_rels[target] = self.vml_drawing_rels_id
            return self.vml_drawing_rels_id

    def _write_font(self, xf_format):
        if False:
            while True:
                i = 10
        xml_writer = self.rstring
        xml_writer._xml_start_tag('rPr')
        if xf_format.bold:
            xml_writer._xml_empty_tag('b')
        if xf_format.italic:
            xml_writer._xml_empty_tag('i')
        if xf_format.font_strikeout:
            xml_writer._xml_empty_tag('strike')
        if xf_format.font_outline:
            xml_writer._xml_empty_tag('outline')
        if xf_format.font_shadow:
            xml_writer._xml_empty_tag('shadow')
        if xf_format.underline:
            self._write_underline(xf_format.underline)
        if xf_format.font_script == 1:
            self._write_vert_align('superscript')
        if xf_format.font_script == 2:
            self._write_vert_align('subscript')
        xml_writer._xml_empty_tag('sz', [('val', xf_format.font_size)])
        if xf_format.theme == -1:
            pass
        elif xf_format.theme:
            self._write_color('theme', xf_format.theme)
        elif xf_format.color_indexed:
            self._write_color('indexed', xf_format.color_indexed)
        elif xf_format.font_color:
            color = self._get_palette_color(xf_format.font_color)
            self._write_rstring_color('rgb', color)
        else:
            self._write_rstring_color('theme', 1)
        xml_writer._xml_empty_tag('rFont', [('val', xf_format.font_name)])
        xml_writer._xml_empty_tag('family', [('val', xf_format.font_family)])
        if xf_format.font_name == 'Calibri' and (not xf_format.hyperlink):
            xml_writer._xml_empty_tag('scheme', [('val', xf_format.font_scheme)])
        xml_writer._xml_end_tag('rPr')

    def _write_underline(self, underline):
        if False:
            i = 10
            return i + 15
        attributes = []
        if underline == 2:
            attributes = [('val', 'double')]
        elif underline == 33:
            attributes = [('val', 'singleAccounting')]
        elif underline == 34:
            attributes = [('val', 'doubleAccounting')]
        self.rstring._xml_empty_tag('u', attributes)

    def _write_vert_align(self, val):
        if False:
            for i in range(10):
                print('nop')
        attributes = [('val', val)]
        self.rstring._xml_empty_tag('vertAlign', attributes)

    def _write_rstring_color(self, name, value):
        if False:
            print('Hello World!')
        attributes = [(name, value)]
        self.rstring._xml_empty_tag('color', attributes)

    def _get_palette_color(self, color):
        if False:
            for i in range(10):
                print('nop')
        if color[0] == '#':
            color = color[1:]
        return 'FF' + color.upper()

    def _opt_close(self):
        if False:
            return 10
        if not self.row_data_fh_closed:
            self.row_data_fh.close()
            self.row_data_fh_closed = True

    def _opt_reopen(self):
        if False:
            i = 10
            return i + 15
        if self.row_data_fh_closed:
            filename = self.row_data_filename
            self.row_data_fh = open(filename, mode='a+', encoding='utf-8')
            self.row_data_fh_closed = False
            self.fh = self.row_data_fh

    def _set_icon_props(self, total_icons, user_props=None):
        if False:
            return 10
        props = []
        for _ in range(total_icons):
            props.append({'criteria': False, 'value': 0, 'type': 'percent'})
        if total_icons == 3:
            props[0]['value'] = 67
            props[1]['value'] = 33
        if total_icons == 4:
            props[0]['value'] = 75
            props[1]['value'] = 50
            props[2]['value'] = 25
        if total_icons == 5:
            props[0]['value'] = 80
            props[1]['value'] = 60
            props[2]['value'] = 40
            props[3]['value'] = 20
        if user_props:
            max_data = len(user_props)
            if max_data >= total_icons:
                max_data = total_icons - 1
            for i in range(max_data):
                if user_props[i].get('value') is not None:
                    props[i]['value'] = user_props[i]['value']
                    tmp = props[i]['value']
                    if isinstance(tmp, str) and tmp.startswith('='):
                        props[i]['value'] = tmp.lstrip('=')
                if user_props[i].get('type'):
                    valid_types = ('percent', 'percentile', 'number', 'formula')
                    if user_props[i]['type'] not in valid_types:
                        warn("Unknown icon property type '%s' for sub-property 'type' in conditional_format()" % user_props[i]['type'])
                    else:
                        props[i]['type'] = user_props[i]['type']
                        if props[i]['type'] == 'number':
                            props[i]['type'] = 'num'
                criteria = user_props[i].get('criteria')
                if criteria and criteria == '>':
                    props[i]['criteria'] = True
        return props

    def _write_worksheet(self):
        if False:
            print('Hello World!')
        schema = 'http://schemas.openxmlformats.org/'
        xmlns = schema + 'spreadsheetml/2006/main'
        xmlns_r = schema + 'officeDocument/2006/relationships'
        xmlns_mc = schema + 'markup-compatibility/2006'
        ms_schema = 'http://schemas.microsoft.com/'
        xmlns_x14ac = ms_schema + 'office/spreadsheetml/2009/9/ac'
        attributes = [('xmlns', xmlns), ('xmlns:r', xmlns_r)]
        if self.excel_version == 2010:
            attributes.append(('xmlns:mc', xmlns_mc))
            attributes.append(('xmlns:x14ac', xmlns_x14ac))
            attributes.append(('mc:Ignorable', 'x14ac'))
        self._xml_start_tag('worksheet', attributes)

    def _write_dimension(self):
        if False:
            while True:
                i = 10
        if self.dim_rowmin is None and self.dim_colmin is None:
            ref = 'A1'
        elif self.dim_rowmin is None and self.dim_colmin is not None:
            if self.dim_colmin == self.dim_colmax:
                ref = xl_rowcol_to_cell(0, self.dim_colmin)
            else:
                cell_1 = xl_rowcol_to_cell(0, self.dim_colmin)
                cell_2 = xl_rowcol_to_cell(0, self.dim_colmax)
                ref = cell_1 + ':' + cell_2
        elif self.dim_rowmin == self.dim_rowmax and self.dim_colmin == self.dim_colmax:
            ref = xl_rowcol_to_cell(self.dim_rowmin, self.dim_colmin)
        else:
            cell_1 = xl_rowcol_to_cell(self.dim_rowmin, self.dim_colmin)
            cell_2 = xl_rowcol_to_cell(self.dim_rowmax, self.dim_colmax)
            ref = cell_1 + ':' + cell_2
        self._xml_empty_tag('dimension', [('ref', ref)])

    def _write_sheet_views(self):
        if False:
            i = 10
            return i + 15
        self._xml_start_tag('sheetViews')
        self._write_sheet_view()
        self._xml_end_tag('sheetViews')

    def _write_sheet_view(self):
        if False:
            for i in range(10):
                print('nop')
        attributes = []
        if not self.screen_gridlines:
            attributes.append(('showGridLines', 0))
        if self.row_col_headers:
            attributes.append(('showRowColHeaders', 0))
        if not self.show_zeros:
            attributes.append(('showZeros', 0))
        if self.is_right_to_left:
            attributes.append(('rightToLeft', 1))
        if self.selected:
            attributes.append(('tabSelected', 1))
        if not self.outline_on:
            attributes.append(('showOutlineSymbols', 0))
        if self.page_view == 1:
            attributes.append(('view', 'pageLayout'))
        elif self.page_view == 2:
            attributes.append(('view', 'pageBreakPreview'))
        if self.top_left_cell != '':
            attributes.append(('topLeftCell', self.top_left_cell))
        if self.zoom != 100:
            attributes.append(('zoomScale', self.zoom))
            if self.page_view == 0 and self.zoom_scale_normal:
                attributes.append(('zoomScaleNormal', self.zoom))
            if self.page_view == 1:
                attributes.append(('zoomScalePageLayoutView', self.zoom))
            if self.page_view == 2:
                attributes.append(('zoomScaleSheetLayoutView', self.zoom))
        attributes.append(('workbookViewId', 0))
        if self.panes or len(self.selections):
            self._xml_start_tag('sheetView', attributes)
            self._write_panes()
            self._write_selections()
            self._xml_end_tag('sheetView')
        else:
            self._xml_empty_tag('sheetView', attributes)

    def _write_sheet_format_pr(self):
        if False:
            i = 10
            return i + 15
        default_row_height = self.default_row_height
        row_level = self.outline_row_level
        col_level = self.outline_col_level
        attributes = [('defaultRowHeight', default_row_height)]
        if self.default_row_height != self.original_row_height:
            attributes.append(('customHeight', 1))
        if self.default_row_zeroed:
            attributes.append(('zeroHeight', 1))
        if row_level:
            attributes.append(('outlineLevelRow', row_level))
        if col_level:
            attributes.append(('outlineLevelCol', col_level))
        if self.excel_version == 2010:
            attributes.append(('x14ac:dyDescent', '0.25'))
        self._xml_empty_tag('sheetFormatPr', attributes)

    def _write_cols(self):
        if False:
            return 10
        if not self.col_info:
            return
        self._xml_start_tag('cols')
        first_col = sorted(self.col_info.keys())[0]
        last_col = first_col
        prev_col_options = self.col_info[first_col]
        del self.col_info[first_col]
        deleted_col = first_col
        deleted_col_options = prev_col_options
        for col in sorted(self.col_info.keys()):
            col_options = self.col_info[col]
            if col == last_col + 1 and col_options == prev_col_options:
                last_col = col
            else:
                self._write_col_info(first_col, last_col, prev_col_options)
                first_col = col
                last_col = first_col
                prev_col_options = col_options
        self._write_col_info(first_col, last_col, prev_col_options)
        self.col_info[deleted_col] = deleted_col_options
        self._xml_end_tag('cols')

    def _write_col_info(self, col_min, col_max, col_info):
        if False:
            i = 10
            return i + 15
        (width, cell_format, hidden, level, collapsed, autofit) = col_info
        custom_width = 1
        xf_index = 0
        if cell_format:
            xf_index = cell_format._get_xf_index()
        if width is None:
            if not hidden:
                width = 8.43
                custom_width = 0
            else:
                width = 0
        elif width == 8.43:
            custom_width = 0
        if width > 0:
            max_digit_width = 7
            padding = 5
            if width < 1:
                width = int(int(width * (max_digit_width + padding) + 0.5) / float(max_digit_width) * 256.0) / 256.0
            else:
                width = int((int(width * max_digit_width + 0.5) + padding) / float(max_digit_width) * 256.0) / 256.0
        attributes = [('min', col_min + 1), ('max', col_max + 1), ('width', '%.16g' % width)]
        if xf_index:
            attributes.append(('style', xf_index))
        if hidden:
            attributes.append(('hidden', '1'))
        if autofit:
            attributes.append(('bestFit', '1'))
        if custom_width:
            attributes.append(('customWidth', '1'))
        if level:
            attributes.append(('outlineLevel', level))
        if collapsed:
            attributes.append(('collapsed', '1'))
        self._xml_empty_tag('col', attributes)

    def _write_sheet_data(self):
        if False:
            while True:
                i = 10
        if self.dim_rowmin is None:
            self._xml_empty_tag('sheetData')
        else:
            self._xml_start_tag('sheetData')
            self._write_rows()
            self._xml_end_tag('sheetData')

    def _write_optimized_sheet_data(self):
        if False:
            while True:
                i = 10
        if self.dim_rowmin is None:
            self._xml_empty_tag('sheetData')
        else:
            self._xml_start_tag('sheetData')
            buff_size = 65536
            self.row_data_fh.seek(0)
            data = self.row_data_fh.read(buff_size)
            while data:
                self.fh.write(data)
                data = self.row_data_fh.read(buff_size)
            self.row_data_fh.close()
            os.unlink(self.row_data_filename)
            self._xml_end_tag('sheetData')

    def _write_page_margins(self):
        if False:
            for i in range(10):
                print('nop')
        attributes = [('left', self.margin_left), ('right', self.margin_right), ('top', self.margin_top), ('bottom', self.margin_bottom), ('header', self.margin_header), ('footer', self.margin_footer)]
        self._xml_empty_tag('pageMargins', attributes)

    def _write_page_setup(self):
        if False:
            print('Hello World!')
        attributes = []
        if not self.page_setup_changed:
            return
        if self.paper_size:
            attributes.append(('paperSize', self.paper_size))
        if self.print_scale != 100:
            attributes.append(('scale', self.print_scale))
        if self.fit_page and self.fit_width != 1:
            attributes.append(('fitToWidth', self.fit_width))
        if self.fit_page and self.fit_height != 1:
            attributes.append(('fitToHeight', self.fit_height))
        if self.page_order:
            attributes.append(('pageOrder', 'overThenDown'))
        if self.page_start > 1:
            attributes.append(('firstPageNumber', self.page_start))
        if self.orientation:
            attributes.append(('orientation', 'portrait'))
        else:
            attributes.append(('orientation', 'landscape'))
        if self.black_white:
            attributes.append(('blackAndWhite', '1'))
        if self.page_start != 0:
            attributes.append(('useFirstPageNumber', '1'))
        if self.is_chartsheet:
            if self.horizontal_dpi:
                attributes.append(('horizontalDpi', self.horizontal_dpi))
            if self.vertical_dpi:
                attributes.append(('verticalDpi', self.vertical_dpi))
        else:
            if self.vertical_dpi:
                attributes.append(('verticalDpi', self.vertical_dpi))
            if self.horizontal_dpi:
                attributes.append(('horizontalDpi', self.horizontal_dpi))
        self._xml_empty_tag('pageSetup', attributes)

    def _write_print_options(self):
        if False:
            i = 10
            return i + 15
        attributes = []
        if not self.print_options_changed:
            return
        if self.hcenter:
            attributes.append(('horizontalCentered', 1))
        if self.vcenter:
            attributes.append(('verticalCentered', 1))
        if self.print_headers:
            attributes.append(('headings', 1))
        if self.print_gridlines:
            attributes.append(('gridLines', 1))
        self._xml_empty_tag('printOptions', attributes)

    def _write_header_footer(self):
        if False:
            print('Hello World!')
        attributes = []
        if not self.header_footer_scales:
            attributes.append(('scaleWithDoc', 0))
        if not self.header_footer_aligns:
            attributes.append(('alignWithMargins', 0))
        if self.header_footer_changed:
            self._xml_start_tag('headerFooter', attributes)
            if self.header:
                self._write_odd_header()
            if self.footer:
                self._write_odd_footer()
            self._xml_end_tag('headerFooter')
        elif self.excel2003_style:
            self._xml_empty_tag('headerFooter', attributes)

    def _write_odd_header(self):
        if False:
            return 10
        self._xml_data_element('oddHeader', self.header)

    def _write_odd_footer(self):
        if False:
            print('Hello World!')
        self._xml_data_element('oddFooter', self.footer)

    def _write_rows(self):
        if False:
            for i in range(10):
                print('nop')
        self._calculate_spans()
        for row_num in range(self.dim_rowmin, self.dim_rowmax + 1):
            if row_num in self.set_rows or row_num in self.comments or self.table[row_num]:
                span_index = int(row_num / 16)
                if span_index in self.row_spans:
                    span = self.row_spans[span_index]
                else:
                    span = None
                if self.table[row_num]:
                    if row_num not in self.set_rows:
                        self._write_row(row_num, span)
                    else:
                        self._write_row(row_num, span, self.set_rows[row_num])
                    for col_num in range(self.dim_colmin, self.dim_colmax + 1):
                        if col_num in self.table[row_num]:
                            col_ref = self.table[row_num][col_num]
                            self._write_cell(row_num, col_num, col_ref)
                    self._xml_end_tag('row')
                elif row_num in self.comments:
                    self._write_empty_row(row_num, span, self.set_rows[row_num])
                else:
                    self._write_empty_row(row_num, span, self.set_rows[row_num])

    def _write_single_row(self, current_row_num=0):
        if False:
            while True:
                i = 10
        row_num = self.previous_row
        self.previous_row = current_row_num
        if row_num in self.set_rows or row_num in self.comments or self.table[row_num]:
            span = None
            if self.table[row_num]:
                if row_num not in self.set_rows:
                    self._write_row(row_num, span)
                else:
                    self._write_row(row_num, span, self.set_rows[row_num])
                for col_num in range(self.dim_colmin, self.dim_colmax + 1):
                    if col_num in self.table[row_num]:
                        col_ref = self.table[row_num][col_num]
                        self._write_cell(row_num, col_num, col_ref)
                self._xml_end_tag('row')
            else:
                self._write_empty_row(row_num, span, self.set_rows[row_num])
        self.table.clear()

    def _calculate_spans(self):
        if False:
            return 10
        spans = {}
        span_min = None
        span_max = None
        for row_num in range(self.dim_rowmin, self.dim_rowmax + 1):
            if row_num in self.table:
                for col_num in range(self.dim_colmin, self.dim_colmax + 1):
                    if col_num in self.table[row_num]:
                        if span_min is None:
                            span_min = col_num
                            span_max = col_num
                        else:
                            if col_num < span_min:
                                span_min = col_num
                            if col_num > span_max:
                                span_max = col_num
            if row_num in self.comments:
                for col_num in range(self.dim_colmin, self.dim_colmax + 1):
                    if row_num in self.comments and col_num in self.comments[row_num]:
                        if span_min is None:
                            span_min = col_num
                            span_max = col_num
                        else:
                            if col_num < span_min:
                                span_min = col_num
                            if col_num > span_max:
                                span_max = col_num
            if (row_num + 1) % 16 == 0 or row_num == self.dim_rowmax:
                span_index = int(row_num / 16)
                if span_min is not None:
                    span_min += 1
                    span_max += 1
                    spans[span_index] = '%s:%s' % (span_min, span_max)
                    span_min = None
        self.row_spans = spans

    def _write_row(self, row, spans, properties=None, empty_row=False):
        if False:
            while True:
                i = 10
        xf_index = 0
        if properties:
            (height, cell_format, hidden, level, collapsed) = properties
        else:
            (height, cell_format, hidden, level, collapsed) = (None, None, 0, 0, 0)
        if height is None:
            height = self.default_row_height
        attributes = [('r', row + 1)]
        if cell_format:
            xf_index = cell_format._get_xf_index()
        if spans:
            attributes.append(('spans', spans))
        if xf_index:
            attributes.append(('s', xf_index))
        if cell_format:
            attributes.append(('customFormat', 1))
        if height != self.original_row_height:
            attributes.append(('ht', '%g' % height))
        if hidden:
            attributes.append(('hidden', 1))
        if height != self.original_row_height:
            attributes.append(('customHeight', 1))
        if level:
            attributes.append(('outlineLevel', level))
        if collapsed:
            attributes.append(('collapsed', 1))
        if self.excel_version == 2010:
            attributes.append(('x14ac:dyDescent', '0.25'))
        if empty_row:
            self._xml_empty_tag_unencoded('row', attributes)
        else:
            self._xml_start_tag_unencoded('row', attributes)

    def _write_empty_row(self, row, spans, properties=None):
        if False:
            i = 10
            return i + 15
        self._write_row(row, spans, properties, empty_row=True)

    def _write_cell(self, row, col, cell):
        if False:
            return 10
        cell_range = xl_rowcol_to_cell_fast(row, col)
        attributes = [('r', cell_range)]
        if cell.format:
            xf_index = cell.format._get_xf_index()
            attributes.append(('s', xf_index))
        elif row in self.set_rows and self.set_rows[row][1]:
            row_xf = self.set_rows[row][1]
            attributes.append(('s', row_xf._get_xf_index()))
        elif col in self.col_info:
            col_xf = self.col_info[col][1]
            if col_xf is not None:
                attributes.append(('s', col_xf._get_xf_index()))
        type_cell_name = cell.__class__.__name__
        if type_cell_name in ('Number', 'Datetime'):
            self._xml_number_element(cell.number, attributes)
        elif type_cell_name in ('String', 'RichString'):
            string = cell.string
            if not self.constant_memory:
                self._xml_string_element(string, attributes)
            else:
                string = self._escape_control_characters(string)
                if string.startswith('<r>') and string.endswith('</r>'):
                    self._xml_rich_inline_string(string, attributes)
                else:
                    preserve = preserve_whitespace(string)
                    self._xml_inline_string(string, preserve, attributes)
        elif type_cell_name == 'Formula':
            value = cell.value
            if isinstance(cell.value, bool):
                attributes.append(('t', 'b'))
                if cell.value:
                    value = 1
                else:
                    value = 0
            elif isinstance(cell.value, str):
                error_codes = ('#DIV/0!', '#N/A', '#NAME?', '#NULL!', '#NUM!', '#REF!', '#VALUE!')
                if cell.value == '':
                    pass
                elif cell.value in error_codes:
                    attributes.append(('t', 'e'))
                else:
                    attributes.append(('t', 'str'))
            self._xml_formula_element(cell.formula, value, attributes)
        elif type_cell_name == 'ArrayFormula':
            if cell.atype == 'dynamic':
                attributes.append(('cm', 1))
            try:
                float(cell.value)
            except ValueError:
                attributes.append(('t', 'str'))
            self._xml_start_tag('c', attributes)
            self._write_cell_array_formula(cell.formula, cell.range)
            self._write_cell_value(cell.value)
            self._xml_end_tag('c')
        elif type_cell_name == 'Blank':
            self._xml_empty_tag('c', attributes)
        elif type_cell_name == 'Boolean':
            attributes.append(('t', 'b'))
            self._xml_start_tag('c', attributes)
            self._write_cell_value(cell.boolean)
            self._xml_end_tag('c')

    def _write_cell_value(self, value):
        if False:
            print('Hello World!')
        if value is None:
            value = ''
        self._xml_data_element('v', value)

    def _write_cell_array_formula(self, formula, cell_range):
        if False:
            for i in range(10):
                print('nop')
        attributes = [('t', 'array'), ('ref', cell_range)]
        self._xml_data_element('f', formula, attributes)

    def _write_sheet_pr(self):
        if False:
            return 10
        attributes = []
        if not self.fit_page and (not self.filter_on) and (not self.tab_color) and (not self.outline_changed) and (not self.vba_codename):
            return
        if self.vba_codename:
            attributes.append(('codeName', self.vba_codename))
        if self.filter_on:
            attributes.append(('filterMode', 1))
        if self.fit_page or self.tab_color or self.outline_changed:
            self._xml_start_tag('sheetPr', attributes)
            self._write_tab_color()
            self._write_outline_pr()
            self._write_page_set_up_pr()
            self._xml_end_tag('sheetPr')
        else:
            self._xml_empty_tag('sheetPr', attributes)

    def _write_page_set_up_pr(self):
        if False:
            while True:
                i = 10
        if not self.fit_page:
            return
        attributes = [('fitToPage', 1)]
        self._xml_empty_tag('pageSetUpPr', attributes)

    def _write_tab_color(self):
        if False:
            return 10
        color = self.tab_color
        if not color:
            return
        attributes = [('rgb', color)]
        self._xml_empty_tag('tabColor', attributes)

    def _write_outline_pr(self):
        if False:
            print('Hello World!')
        attributes = []
        if not self.outline_changed:
            return
        if self.outline_style:
            attributes.append(('applyStyles', 1))
        if not self.outline_below:
            attributes.append(('summaryBelow', 0))
        if not self.outline_right:
            attributes.append(('summaryRight', 0))
        if not self.outline_on:
            attributes.append(('showOutlineSymbols', 0))
        self._xml_empty_tag('outlinePr', attributes)

    def _write_row_breaks(self):
        if False:
            i = 10
            return i + 15
        page_breaks = self._sort_pagebreaks(self.hbreaks)
        if not page_breaks:
            return
        count = len(page_breaks)
        attributes = [('count', count), ('manualBreakCount', count)]
        self._xml_start_tag('rowBreaks', attributes)
        for row_num in page_breaks:
            self._write_brk(row_num, 16383)
        self._xml_end_tag('rowBreaks')

    def _write_col_breaks(self):
        if False:
            for i in range(10):
                print('nop')
        page_breaks = self._sort_pagebreaks(self.vbreaks)
        if not page_breaks:
            return
        count = len(page_breaks)
        attributes = [('count', count), ('manualBreakCount', count)]
        self._xml_start_tag('colBreaks', attributes)
        for col_num in page_breaks:
            self._write_brk(col_num, 1048575)
        self._xml_end_tag('colBreaks')

    def _write_brk(self, brk_id, brk_max):
        if False:
            for i in range(10):
                print('nop')
        attributes = [('id', brk_id), ('max', brk_max), ('man', 1)]
        self._xml_empty_tag('brk', attributes)

    def _write_merge_cells(self):
        if False:
            print('Hello World!')
        merged_cells = self.merge
        count = len(merged_cells)
        if not count:
            return
        attributes = [('count', count)]
        self._xml_start_tag('mergeCells', attributes)
        for merged_range in merged_cells:
            self._write_merge_cell(merged_range)
        self._xml_end_tag('mergeCells')

    def _write_merge_cell(self, merged_range):
        if False:
            return 10
        (row_min, col_min, row_max, col_max) = merged_range
        cell_1 = xl_rowcol_to_cell(row_min, col_min)
        cell_2 = xl_rowcol_to_cell(row_max, col_max)
        ref = cell_1 + ':' + cell_2
        attributes = [('ref', ref)]
        self._xml_empty_tag('mergeCell', attributes)

    def _write_hyperlinks(self):
        if False:
            while True:
                i = 10
        hlink_refs = []
        display = None
        row_nums = sorted(self.hyperlinks.keys())
        if not row_nums:
            return
        for row_num in row_nums:
            col_nums = sorted(self.hyperlinks[row_num].keys())
            for col_num in col_nums:
                link = self.hyperlinks[row_num][col_num]
                link_type = link['link_type']
                if self.table and self.table[row_num] and self.table[row_num][col_num]:
                    cell = self.table[row_num][col_num]
                    if cell.__class__.__name__ != 'String':
                        display = link['url']
                if link_type == 1:
                    self.rel_count += 1
                    hlink_refs.append([link_type, row_num, col_num, self.rel_count, link['str'], display, link['tip']])
                    self.external_hyper_links.append(['/hyperlink', link['url'], 'External'])
                else:
                    hlink_refs.append([link_type, row_num, col_num, link['url'], link['str'], link['tip']])
        self._xml_start_tag('hyperlinks')
        for args in hlink_refs:
            link_type = args.pop(0)
            if link_type == 1:
                self._write_hyperlink_external(*args)
            elif link_type == 2:
                self._write_hyperlink_internal(*args)
        self._xml_end_tag('hyperlinks')

    def _write_hyperlink_external(self, row, col, id_num, location=None, display=None, tooltip=None):
        if False:
            for i in range(10):
                print('nop')
        ref = xl_rowcol_to_cell(row, col)
        r_id = 'rId' + str(id_num)
        attributes = [('ref', ref), ('r:id', r_id)]
        if location is not None:
            attributes.append(('location', location))
        if display is not None:
            attributes.append(('display', display))
        if tooltip is not None:
            attributes.append(('tooltip', tooltip))
        self._xml_empty_tag('hyperlink', attributes)

    def _write_hyperlink_internal(self, row, col, location=None, display=None, tooltip=None):
        if False:
            return 10
        ref = xl_rowcol_to_cell(row, col)
        attributes = [('ref', ref), ('location', location)]
        if tooltip is not None:
            attributes.append(('tooltip', tooltip))
        attributes.append(('display', display))
        self._xml_empty_tag('hyperlink', attributes)

    def _write_auto_filter(self):
        if False:
            i = 10
            return i + 15
        if not self.autofilter_ref:
            return
        attributes = [('ref', self.autofilter_ref)]
        if self.filter_on:
            self._xml_start_tag('autoFilter', attributes)
            self._write_autofilters()
            self._xml_end_tag('autoFilter')
        else:
            self._xml_empty_tag('autoFilter', attributes)

    def _write_autofilters(self):
        if False:
            i = 10
            return i + 15
        (col1, col2) = self.filter_range
        for col in range(col1, col2 + 1):
            if col not in self.filter_cols:
                continue
            tokens = self.filter_cols[col]
            filter_type = self.filter_type[col]
            self._write_filter_column(col - col1, filter_type, tokens)

    def _write_filter_column(self, col_id, filter_type, filters):
        if False:
            i = 10
            return i + 15
        attributes = [('colId', col_id)]
        self._xml_start_tag('filterColumn', attributes)
        if filter_type == 1:
            self._write_filters(filters)
        else:
            self._write_custom_filters(filters)
        self._xml_end_tag('filterColumn')

    def _write_filters(self, filters):
        if False:
            i = 10
            return i + 15
        non_blanks = [filter for filter in filters if str(filter).lower() != 'blanks']
        attributes = []
        if len(filters) != len(non_blanks):
            attributes = [('blank', 1)]
        if len(filters) == 1 and len(non_blanks) == 0:
            self._xml_empty_tag('filters', attributes)
        else:
            self._xml_start_tag('filters', attributes)
            for autofilter in sorted(non_blanks):
                self._write_filter(autofilter)
            self._xml_end_tag('filters')

    def _write_filter(self, val):
        if False:
            return 10
        attributes = [('val', val)]
        self._xml_empty_tag('filter', attributes)

    def _write_custom_filters(self, tokens):
        if False:
            print('Hello World!')
        if len(tokens) == 2:
            self._xml_start_tag('customFilters')
            self._write_custom_filter(*tokens)
            self._xml_end_tag('customFilters')
        else:
            attributes = []
            if tokens[2] == 0:
                attributes = [('and', 1)]
            else:
                attributes = [('and', 0)]
            self._xml_start_tag('customFilters', attributes)
            self._write_custom_filter(tokens[0], tokens[1])
            self._write_custom_filter(tokens[3], tokens[4])
            self._xml_end_tag('customFilters')

    def _write_custom_filter(self, operator, val):
        if False:
            return 10
        attributes = []
        operators = {1: 'lessThan', 2: 'equal', 3: 'lessThanOrEqual', 4: 'greaterThan', 5: 'notEqual', 6: 'greaterThanOrEqual', 22: 'equal'}
        if operators[operator] is not None:
            operator = operators[operator]
        else:
            warn('Unknown operator = %s' % operator)
        if operator != 'equal':
            attributes.append(('operator', operator))
        attributes.append(('val', val))
        self._xml_empty_tag('customFilter', attributes)

    def _write_sheet_protection(self):
        if False:
            return 10
        attributes = []
        if not self.protect_options:
            return
        options = self.protect_options
        if options['password']:
            attributes.append(('password', options['password']))
        if options['sheet']:
            attributes.append(('sheet', 1))
        if options['content']:
            attributes.append(('content', 1))
        if not options['objects']:
            attributes.append(('objects', 1))
        if not options['scenarios']:
            attributes.append(('scenarios', 1))
        if options['format_cells']:
            attributes.append(('formatCells', 0))
        if options['format_columns']:
            attributes.append(('formatColumns', 0))
        if options['format_rows']:
            attributes.append(('formatRows', 0))
        if options['insert_columns']:
            attributes.append(('insertColumns', 0))
        if options['insert_rows']:
            attributes.append(('insertRows', 0))
        if options['insert_hyperlinks']:
            attributes.append(('insertHyperlinks', 0))
        if options['delete_columns']:
            attributes.append(('deleteColumns', 0))
        if options['delete_rows']:
            attributes.append(('deleteRows', 0))
        if not options['select_locked_cells']:
            attributes.append(('selectLockedCells', 1))
        if options['sort']:
            attributes.append(('sort', 0))
        if options['autofilter']:
            attributes.append(('autoFilter', 0))
        if options['pivot_tables']:
            attributes.append(('pivotTables', 0))
        if not options['select_unlocked_cells']:
            attributes.append(('selectUnlockedCells', 1))
        self._xml_empty_tag('sheetProtection', attributes)

    def _write_protected_ranges(self):
        if False:
            for i in range(10):
                print('nop')
        if self.num_protected_ranges == 0:
            return
        self._xml_start_tag('protectedRanges')
        for (cell_range, range_name, password) in self.protected_ranges:
            self._write_protected_range(cell_range, range_name, password)
        self._xml_end_tag('protectedRanges')

    def _write_protected_range(self, cell_range, range_name, password):
        if False:
            print('Hello World!')
        attributes = []
        if password:
            attributes.append(('password', password))
        attributes.append(('sqref', cell_range))
        attributes.append(('name', range_name))
        self._xml_empty_tag('protectedRange', attributes)

    def _write_drawings(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.drawing:
            return
        self.rel_count += 1
        self._write_drawing(self.rel_count)

    def _write_drawing(self, drawing_id):
        if False:
            for i in range(10):
                print('nop')
        r_id = 'rId' + str(drawing_id)
        attributes = [('r:id', r_id)]
        self._xml_empty_tag('drawing', attributes)

    def _write_legacy_drawing(self):
        if False:
            print('Hello World!')
        if not self.has_vml:
            return
        self.rel_count += 1
        r_id = 'rId' + str(self.rel_count)
        attributes = [('r:id', r_id)]
        self._xml_empty_tag('legacyDrawing', attributes)

    def _write_legacy_drawing_hf(self):
        if False:
            return 10
        if not self.has_header_vml:
            return
        self.rel_count += 1
        r_id = 'rId' + str(self.rel_count)
        attributes = [('r:id', r_id)]
        self._xml_empty_tag('legacyDrawingHF', attributes)

    def _write_picture(self):
        if False:
            return 10
        if not self.background_image:
            return
        self.rel_count += 1
        r_id = 'rId' + str(self.rel_count)
        attributes = [('r:id', r_id)]
        self._xml_empty_tag('picture', attributes)

    def _write_data_validations(self):
        if False:
            while True:
                i = 10
        validations = self.validations
        count = len(validations)
        if not count:
            return
        attributes = [('count', count)]
        self._xml_start_tag('dataValidations', attributes)
        for validation in validations:
            self._write_data_validation(validation)
        self._xml_end_tag('dataValidations')

    def _write_data_validation(self, options):
        if False:
            print('Hello World!')
        sqref = ''
        attributes = []
        for cells in options['cells']:
            if sqref != '':
                sqref += ' '
            (row_first, col_first, row_last, col_last) = cells
            if row_first > row_last:
                (row_first, row_last) = (row_last, row_first)
            if col_first > col_last:
                (col_first, col_last) = (col_last, col_first)
            sqref += xl_range(row_first, col_first, row_last, col_last)
        if options.get('multi_range'):
            sqref = options['multi_range']
        if options['validate'] != 'none':
            attributes.append(('type', options['validate']))
            if options['criteria'] != 'between':
                attributes.append(('operator', options['criteria']))
        if 'error_type' in options:
            if options['error_type'] == 1:
                attributes.append(('errorStyle', 'warning'))
            if options['error_type'] == 2:
                attributes.append(('errorStyle', 'information'))
        if options['ignore_blank']:
            attributes.append(('allowBlank', 1))
        if not options['dropdown']:
            attributes.append(('showDropDown', 1))
        if options['show_input']:
            attributes.append(('showInputMessage', 1))
        if options['show_error']:
            attributes.append(('showErrorMessage', 1))
        if 'error_title' in options:
            attributes.append(('errorTitle', options['error_title']))
        if 'error_message' in options:
            attributes.append(('error', options['error_message']))
        if 'input_title' in options:
            attributes.append(('promptTitle', options['input_title']))
        if 'input_message' in options:
            attributes.append(('prompt', options['input_message']))
        attributes.append(('sqref', sqref))
        if options['validate'] == 'none':
            self._xml_empty_tag('dataValidation', attributes)
        else:
            self._xml_start_tag('dataValidation', attributes)
            self._write_formula_1(options['value'])
            if options['maximum'] is not None:
                self._write_formula_2(options['maximum'])
            self._xml_end_tag('dataValidation')

    def _write_formula_1(self, formula):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(formula, list):
            formula = self._csv_join(*formula)
            formula = '"%s"' % formula
        else:
            try:
                float(formula)
            except ValueError:
                if formula.startswith('='):
                    formula = formula.lstrip('=')
        self._xml_data_element('formula1', formula)

    def _write_formula_2(self, formula):
        if False:
            print('Hello World!')
        try:
            float(formula)
        except ValueError:
            if formula.startswith('='):
                formula = formula.lstrip('=')
        self._xml_data_element('formula2', formula)

    def _write_conditional_formats(self):
        if False:
            return 10
        ranges = sorted(self.cond_formats.keys())
        if not ranges:
            return
        for cond_range in ranges:
            self._write_conditional_formatting(cond_range, self.cond_formats[cond_range])

    def _write_conditional_formatting(self, cond_range, params):
        if False:
            print('Hello World!')
        attributes = [('sqref', cond_range)]
        self._xml_start_tag('conditionalFormatting', attributes)
        for param in params:
            self._write_cf_rule(param)
        self._xml_end_tag('conditionalFormatting')

    def _write_cf_rule(self, params):
        if False:
            print('Hello World!')
        attributes = [('type', params['type'])]
        if 'format' in params and params['format'] is not None:
            attributes.append(('dxfId', params['format']))
        attributes.append(('priority', params['priority']))
        if params.get('stop_if_true'):
            attributes.append(('stopIfTrue', 1))
        if params['type'] == 'cellIs':
            attributes.append(('operator', params['criteria']))
            self._xml_start_tag('cfRule', attributes)
            if 'minimum' in params and 'maximum' in params:
                self._write_formula_element(params['minimum'])
                self._write_formula_element(params['maximum'])
            else:
                self._write_formula_element(params['value'])
            self._xml_end_tag('cfRule')
        elif params['type'] == 'aboveAverage':
            if re.search('below', params['criteria']):
                attributes.append(('aboveAverage', 0))
            if re.search('equal', params['criteria']):
                attributes.append(('equalAverage', 1))
            if re.search('[123] std dev', params['criteria']):
                match = re.search('([123]) std dev', params['criteria'])
                attributes.append(('stdDev', match.group(1)))
            self._xml_empty_tag('cfRule', attributes)
        elif params['type'] == 'top10':
            if 'criteria' in params and params['criteria'] == '%':
                attributes.append(('percent', 1))
            if 'direction' in params:
                attributes.append(('bottom', 1))
            rank = params['value'] or 10
            attributes.append(('rank', rank))
            self._xml_empty_tag('cfRule', attributes)
        elif params['type'] == 'duplicateValues':
            self._xml_empty_tag('cfRule', attributes)
        elif params['type'] == 'uniqueValues':
            self._xml_empty_tag('cfRule', attributes)
        elif params['type'] == 'containsText' or params['type'] == 'notContainsText' or params['type'] == 'beginsWith' or (params['type'] == 'endsWith'):
            attributes.append(('operator', params['criteria']))
            attributes.append(('text', params['value']))
            self._xml_start_tag('cfRule', attributes)
            self._write_formula_element(params['formula'])
            self._xml_end_tag('cfRule')
        elif params['type'] == 'timePeriod':
            attributes.append(('timePeriod', params['criteria']))
            self._xml_start_tag('cfRule', attributes)
            self._write_formula_element(params['formula'])
            self._xml_end_tag('cfRule')
        elif params['type'] == 'containsBlanks' or params['type'] == 'notContainsBlanks' or params['type'] == 'containsErrors' or (params['type'] == 'notContainsErrors'):
            self._xml_start_tag('cfRule', attributes)
            self._write_formula_element(params['formula'])
            self._xml_end_tag('cfRule')
        elif params['type'] == 'colorScale':
            self._xml_start_tag('cfRule', attributes)
            self._write_color_scale(params)
            self._xml_end_tag('cfRule')
        elif params['type'] == 'dataBar':
            self._xml_start_tag('cfRule', attributes)
            self._write_data_bar(params)
            if params.get('is_data_bar_2010'):
                self._write_data_bar_ext(params)
            self._xml_end_tag('cfRule')
        elif params['type'] == 'expression':
            self._xml_start_tag('cfRule', attributes)
            self._write_formula_element(params['criteria'])
            self._xml_end_tag('cfRule')
        elif params['type'] == 'iconSet':
            self._xml_start_tag('cfRule', attributes)
            self._write_icon_set(params)
            self._xml_end_tag('cfRule')

    def _write_formula_element(self, formula):
        if False:
            return 10
        try:
            float(formula)
        except ValueError:
            if formula.startswith('='):
                formula = formula.lstrip('=')
        self._xml_data_element('formula', formula)

    def _write_color_scale(self, param):
        if False:
            return 10
        self._xml_start_tag('colorScale')
        self._write_cfvo(param['min_type'], param['min_value'])
        if param['mid_type'] is not None:
            self._write_cfvo(param['mid_type'], param['mid_value'])
        self._write_cfvo(param['max_type'], param['max_value'])
        self._write_color('rgb', param['min_color'])
        if param['mid_color'] is not None:
            self._write_color('rgb', param['mid_color'])
        self._write_color('rgb', param['max_color'])
        self._xml_end_tag('colorScale')

    def _write_data_bar(self, param):
        if False:
            while True:
                i = 10
        attributes = []
        if param.get('min_length'):
            attributes.append(('minLength', param['min_length']))
        if param.get('max_length'):
            attributes.append(('maxLength', param['max_length']))
        if param.get('bar_only'):
            attributes.append(('showValue', 0))
        self._xml_start_tag('dataBar', attributes)
        self._write_cfvo(param['min_type'], param['min_value'])
        self._write_cfvo(param['max_type'], param['max_value'])
        self._write_color('rgb', param['bar_color'])
        self._xml_end_tag('dataBar')

    def _write_data_bar_ext(self, param):
        if False:
            print('Hello World!')
        worksheet_count = self.index + 1
        data_bar_count = len(self.data_bars_2010) + 1
        guid = '{DA7ABA51-AAAA-BBBB-%04X-%012X}' % (worksheet_count, data_bar_count)
        param['guid'] = guid
        self.data_bars_2010.append(param)
        self._xml_start_tag('extLst')
        self._write_ext('{B025F937-C7B1-47D3-B67F-A62EFF666E3E}')
        self._xml_data_element('x14:id', guid)
        self._xml_end_tag('ext')
        self._xml_end_tag('extLst')

    def _write_icon_set(self, param):
        if False:
            for i in range(10):
                print('nop')
        attributes = []
        if param['icon_style'] != '3TrafficLights':
            attributes = [('iconSet', param['icon_style'])]
        if param.get('icons_only'):
            attributes.append(('showValue', 0))
        if param.get('reverse_icons'):
            attributes.append(('reverse', 1))
        self._xml_start_tag('iconSet', attributes)
        for icon in reversed(param['icons']):
            self._write_cfvo(icon['type'], icon['value'], icon['criteria'])
        self._xml_end_tag('iconSet')

    def _write_cfvo(self, cf_type, val, criteria=None):
        if False:
            while True:
                i = 10
        attributes = [('type', cf_type)]
        if val is not None:
            attributes.append(('val', val))
        if criteria:
            attributes.append(('gte', 0))
        self._xml_empty_tag('cfvo', attributes)

    def _write_color(self, name, value):
        if False:
            while True:
                i = 10
        attributes = [(name, value)]
        self._xml_empty_tag('color', attributes)

    def _write_selections(self):
        if False:
            print('Hello World!')
        for selection in self.selections:
            self._write_selection(*selection)

    def _write_selection(self, pane, active_cell, sqref):
        if False:
            i = 10
            return i + 15
        attributes = []
        if pane:
            attributes.append(('pane', pane))
        if active_cell:
            attributes.append(('activeCell', active_cell))
        if sqref:
            attributes.append(('sqref', sqref))
        self._xml_empty_tag('selection', attributes)

    def _write_panes(self):
        if False:
            while True:
                i = 10
        panes = self.panes
        if not len(panes):
            return
        if panes[4] == 2:
            self._write_split_panes(*panes)
        else:
            self._write_freeze_panes(*panes)

    def _write_freeze_panes(self, row, col, top_row, left_col, pane_type):
        if False:
            print('Hello World!')
        attributes = []
        y_split = row
        x_split = col
        top_left_cell = xl_rowcol_to_cell(top_row, left_col)
        active_pane = ''
        state = ''
        active_cell = ''
        sqref = ''
        if self.selections:
            (_, active_cell, sqref) = self.selections[0]
            self.selections = []
        if row and col:
            active_pane = 'bottomRight'
            row_cell = xl_rowcol_to_cell(row, 0)
            col_cell = xl_rowcol_to_cell(0, col)
            self.selections.append(['topRight', col_cell, col_cell])
            self.selections.append(['bottomLeft', row_cell, row_cell])
            self.selections.append(['bottomRight', active_cell, sqref])
        elif col:
            active_pane = 'topRight'
            self.selections.append(['topRight', active_cell, sqref])
        else:
            active_pane = 'bottomLeft'
            self.selections.append(['bottomLeft', active_cell, sqref])
        if pane_type == 0:
            state = 'frozen'
        elif pane_type == 1:
            state = 'frozenSplit'
        else:
            state = 'split'
        if x_split:
            attributes.append(('xSplit', x_split))
        if y_split:
            attributes.append(('ySplit', y_split))
        attributes.append(('topLeftCell', top_left_cell))
        attributes.append(('activePane', active_pane))
        attributes.append(('state', state))
        self._xml_empty_tag('pane', attributes)

    def _write_split_panes(self, row, col, top_row, left_col, pane_type):
        if False:
            return 10
        attributes = []
        has_selection = 0
        active_pane = ''
        active_cell = ''
        sqref = ''
        y_split = row
        x_split = col
        if self.selections:
            (_, active_cell, sqref) = self.selections[0]
            self.selections = []
            has_selection = 1
        if y_split:
            y_split = int(20 * y_split + 300)
        if x_split:
            x_split = self._calculate_x_split_width(x_split)
        if top_row == row and left_col == col:
            top_row = int(0.5 + (y_split - 300) / 20 / 15)
            left_col = int(0.5 + (x_split - 390) / 20 / 3 * 4 / 64)
        top_left_cell = xl_rowcol_to_cell(top_row, left_col)
        if not has_selection:
            active_cell = top_left_cell
            sqref = top_left_cell
        if row and col:
            active_pane = 'bottomRight'
            row_cell = xl_rowcol_to_cell(top_row, 0)
            col_cell = xl_rowcol_to_cell(0, left_col)
            self.selections.append(['topRight', col_cell, col_cell])
            self.selections.append(['bottomLeft', row_cell, row_cell])
            self.selections.append(['bottomRight', active_cell, sqref])
        elif col:
            active_pane = 'topRight'
            self.selections.append(['topRight', active_cell, sqref])
        else:
            active_pane = 'bottomLeft'
            self.selections.append(['bottomLeft', active_cell, sqref])
        if x_split:
            attributes.append(('xSplit', '%.16g' % x_split))
        if y_split:
            attributes.append(('ySplit', '%.16g' % y_split))
        attributes.append(('topLeftCell', top_left_cell))
        if has_selection:
            attributes.append(('activePane', active_pane))
        self._xml_empty_tag('pane', attributes)

    def _calculate_x_split_width(self, width):
        if False:
            i = 10
            return i + 15
        max_digit_width = 7
        padding = 5
        if width < 1:
            pixels = int(width * (max_digit_width + padding) + 0.5)
        else:
            pixels = int(width * max_digit_width + 0.5) + padding
        points = pixels * 3 / 4
        twips = points * 20
        width = twips + 390
        return width

    def _write_table_parts(self):
        if False:
            return 10
        tables = self.tables
        count = len(tables)
        if not count:
            return
        attributes = [('count', count)]
        self._xml_start_tag('tableParts', attributes)
        for _ in tables:
            self.rel_count += 1
            self._write_table_part(self.rel_count)
        self._xml_end_tag('tableParts')

    def _write_table_part(self, r_id):
        if False:
            while True:
                i = 10
        r_id = 'rId' + str(r_id)
        attributes = [('r:id', r_id)]
        self._xml_empty_tag('tablePart', attributes)

    def _write_ext_list(self):
        if False:
            for i in range(10):
                print('nop')
        has_data_bars = len(self.data_bars_2010)
        has_sparklines = len(self.sparklines)
        if not has_data_bars and (not has_sparklines):
            return
        self._xml_start_tag('extLst')
        if has_data_bars:
            self._write_ext_list_data_bars()
        if has_sparklines:
            self._write_ext_list_sparklines()
        self._xml_end_tag('extLst')

    def _write_ext_list_data_bars(self):
        if False:
            i = 10
            return i + 15
        self._write_ext('{78C0D931-6437-407d-A8EE-F0AAD7539E65}')
        self._xml_start_tag('x14:conditionalFormattings')
        for data_bar in self.data_bars_2010:
            self._write_conditional_formatting_2010(data_bar)
        self._xml_end_tag('x14:conditionalFormattings')
        self._xml_end_tag('ext')

    def _write_conditional_formatting_2010(self, data_bar):
        if False:
            i = 10
            return i + 15
        xmlns_xm = 'http://schemas.microsoft.com/office/excel/2006/main'
        attributes = [('xmlns:xm', xmlns_xm)]
        self._xml_start_tag('x14:conditionalFormatting', attributes)
        self._write_x14_cf_rule(data_bar)
        self._write_x14_data_bar(data_bar)
        self._write_x14_cfvo(data_bar['x14_min_type'], data_bar['min_value'])
        self._write_x14_cfvo(data_bar['x14_max_type'], data_bar['max_value'])
        if not data_bar['bar_no_border']:
            self._write_x14_border_color(data_bar['bar_border_color'])
        if not data_bar['bar_negative_color_same']:
            self._write_x14_negative_fill_color(data_bar['bar_negative_color'])
        if not data_bar['bar_no_border'] and (not data_bar['bar_negative_border_color_same']):
            self._write_x14_negative_border_color(data_bar['bar_negative_border_color'])
        if data_bar['bar_axis_position'] != 'none':
            self._write_x14_axis_color(data_bar['bar_axis_color'])
        self._xml_end_tag('x14:dataBar')
        self._xml_end_tag('x14:cfRule')
        self._xml_data_element('xm:sqref', data_bar['range'])
        self._xml_end_tag('x14:conditionalFormatting')

    def _write_x14_cf_rule(self, data_bar):
        if False:
            for i in range(10):
                print('nop')
        rule_type = 'dataBar'
        guid = data_bar['guid']
        attributes = [('type', rule_type), ('id', guid)]
        self._xml_start_tag('x14:cfRule', attributes)

    def _write_x14_data_bar(self, data_bar):
        if False:
            i = 10
            return i + 15
        min_length = 0
        max_length = 100
        attributes = [('minLength', min_length), ('maxLength', max_length)]
        if not data_bar['bar_no_border']:
            attributes.append(('border', 1))
        if data_bar['bar_solid']:
            attributes.append(('gradient', 0))
        if data_bar['bar_direction'] == 'left':
            attributes.append(('direction', 'leftToRight'))
        if data_bar['bar_direction'] == 'right':
            attributes.append(('direction', 'rightToLeft'))
        if data_bar['bar_negative_color_same']:
            attributes.append(('negativeBarColorSameAsPositive', 1))
        if not data_bar['bar_no_border'] and (not data_bar['bar_negative_border_color_same']):
            attributes.append(('negativeBarBorderColorSameAsPositive', 0))
        if data_bar['bar_axis_position'] == 'middle':
            attributes.append(('axisPosition', 'middle'))
        if data_bar['bar_axis_position'] == 'none':
            attributes.append(('axisPosition', 'none'))
        self._xml_start_tag('x14:dataBar', attributes)

    def _write_x14_cfvo(self, rule_type, value):
        if False:
            print('Hello World!')
        attributes = [('type', rule_type)]
        if rule_type in ('min', 'max', 'autoMin', 'autoMax'):
            self._xml_empty_tag('x14:cfvo', attributes)
        else:
            self._xml_start_tag('x14:cfvo', attributes)
            self._xml_data_element('xm:f', value)
            self._xml_end_tag('x14:cfvo')

    def _write_x14_border_color(self, rgb):
        if False:
            for i in range(10):
                print('nop')
        attributes = [('rgb', rgb)]
        self._xml_empty_tag('x14:borderColor', attributes)

    def _write_x14_negative_fill_color(self, rgb):
        if False:
            print('Hello World!')
        attributes = [('rgb', rgb)]
        self._xml_empty_tag('x14:negativeFillColor', attributes)

    def _write_x14_negative_border_color(self, rgb):
        if False:
            for i in range(10):
                print('nop')
        attributes = [('rgb', rgb)]
        self._xml_empty_tag('x14:negativeBorderColor', attributes)

    def _write_x14_axis_color(self, rgb):
        if False:
            print('Hello World!')
        attributes = [('rgb', rgb)]
        self._xml_empty_tag('x14:axisColor', attributes)

    def _write_ext_list_sparklines(self):
        if False:
            for i in range(10):
                print('nop')
        self._write_ext('{05C60535-1F16-4fd2-B633-F4F36F0B64E0}')
        self._write_sparkline_groups()
        for sparkline in reversed(self.sparklines):
            self._write_sparkline_group(sparkline)
            self._write_color_series(sparkline['series_color'])
            self._write_color_negative(sparkline['negative_color'])
            self._write_color_axis()
            self._write_color_markers(sparkline['markers_color'])
            self._write_color_first(sparkline['first_color'])
            self._write_color_last(sparkline['last_color'])
            self._write_color_high(sparkline['high_color'])
            self._write_color_low(sparkline['low_color'])
            if sparkline['date_axis']:
                self._xml_data_element('xm:f', sparkline['date_axis'])
            self._write_sparklines(sparkline)
            self._xml_end_tag('x14:sparklineGroup')
        self._xml_end_tag('x14:sparklineGroups')
        self._xml_end_tag('ext')

    def _write_sparklines(self, sparkline):
        if False:
            for i in range(10):
                print('nop')
        self._xml_start_tag('x14:sparklines')
        for i in range(sparkline['count']):
            spark_range = sparkline['ranges'][i]
            location = sparkline['locations'][i]
            self._xml_start_tag('x14:sparkline')
            self._xml_data_element('xm:f', spark_range)
            self._xml_data_element('xm:sqref', location)
            self._xml_end_tag('x14:sparkline')
        self._xml_end_tag('x14:sparklines')

    def _write_ext(self, uri):
        if False:
            while True:
                i = 10
        schema = 'http://schemas.microsoft.com/office/'
        xmlns_x14 = schema + 'spreadsheetml/2009/9/main'
        attributes = [('xmlns:x14', xmlns_x14), ('uri', uri)]
        self._xml_start_tag('ext', attributes)

    def _write_sparkline_groups(self):
        if False:
            while True:
                i = 10
        xmlns_xm = 'http://schemas.microsoft.com/office/excel/2006/main'
        attributes = [('xmlns:xm', xmlns_xm)]
        self._xml_start_tag('x14:sparklineGroups', attributes)

    def _write_sparkline_group(self, options):
        if False:
            while True:
                i = 10
        empty = options.get('empty')
        attributes = []
        if options.get('max') is not None:
            if options['max'] == 'group':
                options['cust_max'] = 'group'
            else:
                attributes.append(('manualMax', options['max']))
                options['cust_max'] = 'custom'
        if options.get('min') is not None:
            if options['min'] == 'group':
                options['cust_min'] = 'group'
            else:
                attributes.append(('manualMin', options['min']))
                options['cust_min'] = 'custom'
        if options['type'] != 'line':
            attributes.append(('type', options['type']))
        if options.get('weight'):
            attributes.append(('lineWeight', options['weight']))
        if options.get('date_axis'):
            attributes.append(('dateAxis', 1))
        if empty:
            attributes.append(('displayEmptyCellsAs', empty))
        if options.get('markers'):
            attributes.append(('markers', 1))
        if options.get('high'):
            attributes.append(('high', 1))
        if options.get('low'):
            attributes.append(('low', 1))
        if options.get('first'):
            attributes.append(('first', 1))
        if options.get('last'):
            attributes.append(('last', 1))
        if options.get('negative'):
            attributes.append(('negative', 1))
        if options.get('axis'):
            attributes.append(('displayXAxis', 1))
        if options.get('hidden'):
            attributes.append(('displayHidden', 1))
        if options.get('cust_min'):
            attributes.append(('minAxisType', options['cust_min']))
        if options.get('cust_max'):
            attributes.append(('maxAxisType', options['cust_max']))
        if options.get('reverse'):
            attributes.append(('rightToLeft', 1))
        self._xml_start_tag('x14:sparklineGroup', attributes)

    def _write_spark_color(self, element, color):
        if False:
            return 10
        attributes = []
        if color.get('rgb'):
            attributes.append(('rgb', color['rgb']))
        if color.get('theme'):
            attributes.append(('theme', color['theme']))
        if color.get('tint'):
            attributes.append(('tint', color['tint']))
        self._xml_empty_tag(element, attributes)

    def _write_color_series(self, color):
        if False:
            print('Hello World!')
        self._write_spark_color('x14:colorSeries', color)

    def _write_color_negative(self, color):
        if False:
            return 10
        self._write_spark_color('x14:colorNegative', color)

    def _write_color_axis(self):
        if False:
            i = 10
            return i + 15
        self._write_spark_color('x14:colorAxis', {'rgb': 'FF000000'})

    def _write_color_markers(self, color):
        if False:
            while True:
                i = 10
        self._write_spark_color('x14:colorMarkers', color)

    def _write_color_first(self, color):
        if False:
            i = 10
            return i + 15
        self._write_spark_color('x14:colorFirst', color)

    def _write_color_last(self, color):
        if False:
            return 10
        self._write_spark_color('x14:colorLast', color)

    def _write_color_high(self, color):
        if False:
            for i in range(10):
                print('nop')
        self._write_spark_color('x14:colorHigh', color)

    def _write_color_low(self, color):
        if False:
            for i in range(10):
                print('nop')
        self._write_spark_color('x14:colorLow', color)

    def _write_phonetic_pr(self):
        if False:
            for i in range(10):
                print('nop')
        attributes = [('fontId', '0'), ('type', 'noConversion')]
        self._xml_empty_tag('phoneticPr', attributes)

    def _write_ignored_errors(self):
        if False:
            return 10
        if not self.ignored_errors:
            return
        self._xml_start_tag('ignoredErrors')
        if self.ignored_errors.get('number_stored_as_text'):
            ignored_range = self.ignored_errors['number_stored_as_text']
            self._write_ignored_error('numberStoredAsText', ignored_range)
        if self.ignored_errors.get('eval_error'):
            ignored_range = self.ignored_errors['eval_error']
            self._write_ignored_error('evalError', ignored_range)
        if self.ignored_errors.get('formula_differs'):
            ignored_range = self.ignored_errors['formula_differs']
            self._write_ignored_error('formula', ignored_range)
        if self.ignored_errors.get('formula_range'):
            ignored_range = self.ignored_errors['formula_range']
            self._write_ignored_error('formulaRange', ignored_range)
        if self.ignored_errors.get('formula_unlocked'):
            ignored_range = self.ignored_errors['formula_unlocked']
            self._write_ignored_error('unlockedFormula', ignored_range)
        if self.ignored_errors.get('empty_cell_reference'):
            ignored_range = self.ignored_errors['empty_cell_reference']
            self._write_ignored_error('emptyCellReference', ignored_range)
        if self.ignored_errors.get('list_data_validation'):
            ignored_range = self.ignored_errors['list_data_validation']
            self._write_ignored_error('listDataValidation', ignored_range)
        if self.ignored_errors.get('calculated_column'):
            ignored_range = self.ignored_errors['calculated_column']
            self._write_ignored_error('calculatedColumn', ignored_range)
        if self.ignored_errors.get('two_digit_text_year'):
            ignored_range = self.ignored_errors['two_digit_text_year']
            self._write_ignored_error('twoDigitTextYear', ignored_range)
        self._xml_end_tag('ignoredErrors')

    def _write_ignored_error(self, type, ignored_range):
        if False:
            for i in range(10):
                print('nop')
        attributes = [('sqref', ignored_range), (type, 1)]
        self._xml_empty_tag('ignoredError', attributes)