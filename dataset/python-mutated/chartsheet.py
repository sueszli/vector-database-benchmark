from . import worksheet
from .drawing import Drawing

class Chartsheet(worksheet.Worksheet):
    """
    A class for writing the Excel XLSX Chartsheet file.


    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Constructor.\n\n        '
        super(Chartsheet, self).__init__()
        self.is_chartsheet = True
        self.drawing = None
        self.chart = None
        self.charts = []
        self.zoom_scale_normal = 0
        self.orientation = 0
        self.protection = False

    def set_chart(self, chart):
        if False:
            while True:
                i = 10
        '\n        Set the chart object for the chartsheet.\n        Args:\n            chart:  Chart object.\n        Returns:\n            chart:  A reference to the chart object.\n        '
        chart.embedded = False
        chart.protection = self.protection
        self.chart = chart
        self.charts.append([0, 0, chart, 0, 0, 1, 1])
        return chart

    def protect(self, password='', options=None):
        if False:
            i = 10
            return i + 15
        '\n        Set the password and protection options of the worksheet.\n\n        Args:\n            password: An optional password string.\n            options:  A dictionary of worksheet objects to protect.\n\n        Returns:\n            Nothing.\n\n        '
        copy = {}
        if not options:
            options = {}
        if options.get('objects') is None:
            copy['objects'] = False
        else:
            copy['objects'] = not options['objects']
        if options.get('content') is None:
            copy['content'] = True
        else:
            copy['content'] = options['content']
        copy['sheet'] = False
        copy['scenarios'] = True
        if password == '' and copy['objects'] and (not copy['content']):
            return
        if self.chart:
            self.chart.protection = True
        else:
            self.protection = True
        super(Chartsheet, self).protect(password, copy)

    def _assemble_xml_file(self):
        if False:
            while True:
                i = 10
        self._xml_declaration()
        self._write_chartsheet()
        self._write_sheet_pr()
        self._write_sheet_views()
        self._write_sheet_protection()
        self._write_print_options()
        self._write_page_margins()
        self._write_page_setup()
        self._write_header_footer()
        self._write_drawings()
        self._write_legacy_drawing_hf()
        self._xml_end_tag('chartsheet')
        self._xml_close()

    def _prepare_chart(self, index, chart_id, drawing_id):
        if False:
            i = 10
            return i + 15
        self.chart.id = chart_id - 1
        self.drawing = Drawing()
        self.drawing.orientation = self.orientation
        self.external_drawing_links.append(['/drawing', '../drawings/drawing' + str(drawing_id) + '.xml'])
        self.drawing_links.append(['/chart', '../charts/chart' + str(chart_id) + '.xml'])

    def _write_chartsheet(self):
        if False:
            return 10
        schema = 'http://schemas.openxmlformats.org/'
        xmlns = schema + 'spreadsheetml/2006/main'
        xmlns_r = schema + 'officeDocument/2006/relationships'
        attributes = [('xmlns', xmlns), ('xmlns:r', xmlns_r)]
        self._xml_start_tag('chartsheet', attributes)

    def _write_sheet_pr(self):
        if False:
            i = 10
            return i + 15
        attributes = []
        if self.filter_on:
            attributes.append(('filterMode', 1))
        if self.fit_page or self.tab_color:
            self._xml_start_tag('sheetPr', attributes)
            self._write_tab_color()
            self._write_page_set_up_pr()
            self._xml_end_tag('sheetPr')
        else:
            self._xml_empty_tag('sheetPr', attributes)