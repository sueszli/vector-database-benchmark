from os.path import abspath, dirname, join
import astropy.config as _config
import astropy.io.registry as io_registry
from astropy import extern
from .table import Table

class Conf(_config.ConfigNamespace):
    """
    Configuration parameters for `astropy.table.jsviewer`.
    """
    jquery_url = _config.ConfigItem('https://code.jquery.com/jquery-3.6.0.min.js', 'The URL to the jquery library.')
    datatables_url = _config.ConfigItem('https://cdn.datatables.net/1.10.12/js/jquery.dataTables.min.js', 'The URL to the jquery datatables library.')
    css_urls = _config.ConfigItem(['https://cdn.datatables.net/1.10.12/css/jquery.dataTables.css'], 'The URLs to the css file(s) to include.', cfgtype='string_list')
conf = Conf()
EXTERN_JS_DIR = abspath(join(dirname(extern.__file__), 'jquery', 'data', 'js'))
EXTERN_CSS_DIR = abspath(join(dirname(extern.__file__), 'jquery', 'data', 'css'))
_SORTING_SCRIPT_PART_1 = '\nvar astropy_sort_num = function(a, b) {{\n    var a_num = parseFloat(a);\n    var b_num = parseFloat(b);\n\n    if (isNaN(a_num) && isNaN(b_num))\n        return ((a < b) ? -1 : ((a > b) ? 1 : 0));\n    else if (!isNaN(a_num) && !isNaN(b_num))\n        return ((a_num < b_num) ? -1 : ((a_num > b_num) ? 1 : 0));\n    else\n        return isNaN(a_num) ? -1 : 1;\n}}\n'
_SORTING_SCRIPT_PART_2 = '\njQuery.extend( jQuery.fn.dataTableExt.oSort, {{\n    "optionalnum-asc": astropy_sort_num,\n    "optionalnum-desc": function (a,b) {{ return -astropy_sort_num(a, b); }}\n}});\n'
IPYNB_JS_SCRIPT = '\n<script>\n%(sorting_script1)s\nrequire.config({{paths: {{\n    datatables: \'{datatables_url}\'\n}}}});\nrequire(["datatables"], function(){{\n    console.log("$(\'#{tid}\').dataTable()");\n    %(sorting_script2)s\n    $(\'#{tid}\').dataTable({{\n        order: [],\n        pageLength: {display_length},\n        lengthMenu: {display_length_menu},\n        pagingType: "full_numbers",\n        columnDefs: [{{targets: {sort_columns}, type: "optionalnum"}}]\n    }});\n}});\n</script>\n' % dict(sorting_script1=_SORTING_SCRIPT_PART_1, sorting_script2=_SORTING_SCRIPT_PART_2)
HTML_JS_SCRIPT = _SORTING_SCRIPT_PART_1 + _SORTING_SCRIPT_PART_2 + '\n$(document).ready(function() {{\n    $(\'#{tid}\').dataTable({{\n        order: [],\n        pageLength: {display_length},\n        lengthMenu: {display_length_menu},\n        pagingType: "full_numbers",\n        columnDefs: [{{targets: {sort_columns}, type: "optionalnum"}}]\n    }});\n}} );\n'
DEFAULT_CSS = 'body {font-family: sans-serif;}\ntable.dataTable {width: auto !important; margin: 0 !important;}\n.dataTables_filter, .dataTables_paginate {float: left !important; margin-left:1em}\n'
DEFAULT_CSS_NB = 'table.dataTable {clear: both; width: auto !important; margin: 0 !important;}\n.dataTables_info, .dataTables_length, .dataTables_filter, .dataTables_paginate{\ndisplay: inline-block; margin-right: 1em; }\n.paginate_button { margin-right: 5px; }\n'

class JSViewer:
    """Provides an interactive HTML export of a Table.

    This class provides an interface to the `DataTables
    <https://datatables.net/>`_ library, which allow to visualize interactively
    an HTML table. It is used by the `~astropy.table.Table.show_in_browser`
    method.

    Parameters
    ----------
    use_local_files : bool, optional
        Use local files or a CDN for JavaScript libraries. Default False.
    display_length : int, optional
        Number or rows to show. Default to 50.

    """

    def __init__(self, use_local_files=False, display_length=50):
        if False:
            print('Hello World!')
        self._use_local_files = use_local_files
        self.display_length_menu = [[10, 25, 50, 100, 500, 1000, -1], [10, 25, 50, 100, 500, 1000, 'All']]
        self.display_length = display_length
        for L in self.display_length_menu:
            if display_length not in L:
                L.insert(0, display_length)

    @property
    def jquery_urls(self):
        if False:
            while True:
                i = 10
        if self._use_local_files:
            return ['file://' + join(EXTERN_JS_DIR, 'jquery-3.6.0.min.js'), 'file://' + join(EXTERN_JS_DIR, 'jquery.dataTables.min.js')]
        else:
            return [conf.jquery_url, conf.datatables_url]

    @property
    def css_urls(self):
        if False:
            while True:
                i = 10
        if self._use_local_files:
            return ['file://' + join(EXTERN_CSS_DIR, 'jquery.dataTables.css')]
        else:
            return conf.css_urls

    def _jstable_file(self):
        if False:
            for i in range(10):
                print('nop')
        if self._use_local_files:
            return 'file://' + join(EXTERN_JS_DIR, 'jquery.dataTables.min')
        else:
            return conf.datatables_url[:-3]

    def ipynb(self, table_id, css=None, sort_columns='[]'):
        if False:
            return 10
        html = f'<style>{(css if css is not None else DEFAULT_CSS_NB)}</style>'
        html += IPYNB_JS_SCRIPT.format(display_length=self.display_length, display_length_menu=self.display_length_menu, datatables_url=self._jstable_file(), tid=table_id, sort_columns=sort_columns)
        return html

    def html_js(self, table_id='table0', sort_columns='[]'):
        if False:
            return 10
        return HTML_JS_SCRIPT.format(display_length=self.display_length, display_length_menu=self.display_length_menu, tid=table_id, sort_columns=sort_columns).strip()

def write_table_jsviewer(table, filename, table_id=None, max_lines=5000, table_class='display compact', jskwargs=None, css=DEFAULT_CSS, htmldict=None, overwrite=False):
    if False:
        print('Hello World!')
    if table_id is None:
        table_id = f'table{id(table)}'
    jskwargs = jskwargs or {}
    jsv = JSViewer(**jskwargs)
    sortable_columns = [i for (i, col) in enumerate(table.columns.values()) if col.info.dtype.kind in 'iufc']
    html_options = {'table_id': table_id, 'table_class': table_class, 'css': css, 'cssfiles': jsv.css_urls, 'jsfiles': jsv.jquery_urls, 'js': jsv.html_js(table_id=table_id, sort_columns=sortable_columns)}
    if htmldict:
        html_options.update(htmldict)
    if max_lines < len(table):
        table = table[:max_lines]
    table.write(filename, format='html', htmldict=html_options, overwrite=overwrite)
io_registry.register_writer('jsviewer', Table, write_table_jsviewer)