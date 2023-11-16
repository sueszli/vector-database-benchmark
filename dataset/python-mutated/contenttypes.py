import copy
from . import xmlwriter
app_package = 'application/vnd.openxmlformats-package.'
app_document = 'application/vnd.openxmlformats-officedocument.'
defaults = [['rels', app_package + 'relationships+xml'], ['xml', 'application/xml']]
overrides = [['/docProps/app.xml', app_document + 'extended-properties+xml'], ['/docProps/core.xml', app_package + 'core-properties+xml'], ['/xl/styles.xml', app_document + 'spreadsheetml.styles+xml'], ['/xl/theme/theme1.xml', app_document + 'theme+xml'], ['/xl/workbook.xml', app_document + 'spreadsheetml.sheet.main+xml']]

class ContentTypes(xmlwriter.XMLwriter):
    """
    A class for writing the Excel XLSX ContentTypes file.


    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Constructor.\n\n        '
        super(ContentTypes, self).__init__()
        self.defaults = copy.deepcopy(defaults)
        self.overrides = copy.deepcopy(overrides)

    def _assemble_xml_file(self):
        if False:
            print('Hello World!')
        self._xml_declaration()
        self._write_types()
        self._write_defaults()
        self._write_overrides()
        self._xml_end_tag('Types')
        self._xml_close()

    def _add_default(self, default):
        if False:
            return 10
        self.defaults.append(default)

    def _add_override(self, override):
        if False:
            while True:
                i = 10
        self.overrides.append(override)

    def _add_worksheet_name(self, worksheet_name):
        if False:
            print('Hello World!')
        worksheet_name = '/xl/worksheets/' + worksheet_name + '.xml'
        self._add_override((worksheet_name, app_document + 'spreadsheetml.worksheet+xml'))

    def _add_chartsheet_name(self, chartsheet_name):
        if False:
            print('Hello World!')
        chartsheet_name = '/xl/chartsheets/' + chartsheet_name + '.xml'
        self._add_override((chartsheet_name, app_document + 'spreadsheetml.chartsheet+xml'))

    def _add_chart_name(self, chart_name):
        if False:
            while True:
                i = 10
        chart_name = '/xl/charts/' + chart_name + '.xml'
        self._add_override((chart_name, app_document + 'drawingml.chart+xml'))

    def _add_drawing_name(self, drawing_name):
        if False:
            for i in range(10):
                print('nop')
        drawing_name = '/xl/drawings/' + drawing_name + '.xml'
        self._add_override((drawing_name, app_document + 'drawing+xml'))

    def _add_vml_name(self):
        if False:
            while True:
                i = 10
        self._add_default(('vml', app_document + 'vmlDrawing'))

    def _add_comment_name(self, comment_name):
        if False:
            i = 10
            return i + 15
        comment_name = '/xl/' + comment_name + '.xml'
        self._add_override((comment_name, app_document + 'spreadsheetml.comments+xml'))

    def _add_shared_strings(self):
        if False:
            while True:
                i = 10
        self._add_override(('/xl/sharedStrings.xml', app_document + 'spreadsheetml.sharedStrings+xml'))

    def _add_calc_chain(self):
        if False:
            while True:
                i = 10
        self._add_override(('/xl/calcChain.xml', app_document + 'spreadsheetml.calcChain+xml'))

    def _add_image_types(self, image_types):
        if False:
            for i in range(10):
                print('nop')
        for image_type in image_types:
            extension = image_type
            if image_type in ('wmf', 'emf'):
                image_type = 'x-' + image_type
            self._add_default((extension, 'image/' + image_type))

    def _add_table_name(self, table_name):
        if False:
            for i in range(10):
                print('nop')
        table_name = '/xl/tables/' + table_name + '.xml'
        self._add_override((table_name, app_document + 'spreadsheetml.table+xml'))

    def _add_vba_project(self):
        if False:
            return 10
        for (i, override) in enumerate(self.overrides):
            if override[0] == '/xl/workbook.xml':
                xlsm = 'application/vnd.ms-excel.sheet.macroEnabled.main+xml'
                self.overrides[i][1] = xlsm
        self._add_default(('bin', 'application/vnd.ms-office.vbaProject'))

    def _add_vba_project_signature(self):
        if False:
            while True:
                i = 10
        self._add_override(('/xl/vbaProjectSignature.bin', 'application/vnd.ms-office.vbaProjectSignature'))

    def _add_custom_properties(self):
        if False:
            print('Hello World!')
        self._add_override(('/docProps/custom.xml', app_document + 'custom-properties+xml'))

    def _add_metadata(self):
        if False:
            print('Hello World!')
        self._add_override(('/xl/metadata.xml', app_document + 'spreadsheetml.sheetMetadata+xml'))

    def _write_defaults(self):
        if False:
            return 10
        for (extension, content_type) in self.defaults:
            self._xml_empty_tag('Default', [('Extension', extension), ('ContentType', content_type)])

    def _write_overrides(self):
        if False:
            while True:
                i = 10
        for (part_name, content_type) in self.overrides:
            self._xml_empty_tag('Override', [('PartName', part_name), ('ContentType', content_type)])

    def _write_types(self):
        if False:
            i = 10
            return i + 15
        xmlns = 'http://schemas.openxmlformats.org/package/2006/content-types'
        attributes = [('xmlns', xmlns)]
        self._xml_start_tag('Types', attributes)

    def _write_default(self, extension, content_type):
        if False:
            return 10
        attributes = [('Extension', extension), ('ContentType', content_type)]
        self._xml_empty_tag('Default', attributes)

    def _write_override(self, part_name, content_type):
        if False:
            while True:
                i = 10
        attributes = [('PartName', part_name), ('ContentType', content_type)]
        self._xml_empty_tag('Override', attributes)