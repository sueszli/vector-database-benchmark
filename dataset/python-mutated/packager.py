import os
import stat
import tempfile
from shutil import copy
from io import StringIO
from io import BytesIO
from .app import App
from .contenttypes import ContentTypes
from .core import Core
from .custom import Custom
from .metadata import Metadata
from .relationships import Relationships
from .sharedstrings import SharedStrings
from .styles import Styles
from .theme import Theme
from .vml import Vml
from .table import Table
from .comments import Comments
from .exceptions import EmptyChartSeries

class Packager(object):
    """
    A class for writing the Excel XLSX Packager file.

    This module is used in conjunction with XlsxWriter to create an
    Excel XLSX container file.

    From Wikipedia: The Open Packaging Conventions (OPC) is a
    container-file technology initially created by Microsoft to store
    a combination of XML and non-XML files that together form a single
    entity such as an Open XML Paper Specification (OpenXPS)
    document. http://en.wikipedia.org/wiki/Open_Packaging_Conventions.

    At its simplest an Excel XLSX file contains the following elements::

         ____ [Content_Types].xml
        |
        |____ docProps
        | |____ app.xml
        | |____ core.xml
        |
        |____ xl
        | |____ workbook.xml
        | |____ worksheets
        | | |____ sheet1.xml
        | |
        | |____ styles.xml
        | |
        | |____ theme
        | | |____ theme1.xml
        | |
        | |_____rels
        |   |____ workbook.xml.rels
        |
        |_____rels
          |____ .rels

    The Packager class coordinates the classes that represent the
    elements of the package and writes them into the XLSX file.

    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        '\n        Constructor.\n\n        '
        super(Packager, self).__init__()
        self.tmpdir = ''
        self.in_memory = False
        self.workbook = None
        self.worksheet_count = 0
        self.chartsheet_count = 0
        self.chart_count = 0
        self.drawing_count = 0
        self.table_count = 0
        self.num_vml_files = 0
        self.num_comment_files = 0
        self.named_ranges = []
        self.filenames = []

    def _set_tmpdir(self, tmpdir):
        if False:
            for i in range(10):
                print('nop')
        self.tmpdir = tmpdir

    def _set_in_memory(self, in_memory):
        if False:
            i = 10
            return i + 15
        self.in_memory = in_memory

    def _add_workbook(self, workbook):
        if False:
            while True:
                i = 10
        self.workbook = workbook
        self.chart_count = len(workbook.charts)
        self.drawing_count = len(workbook.drawings)
        self.num_vml_files = workbook.num_vml_files
        self.num_comment_files = workbook.num_comment_files
        self.named_ranges = workbook.named_ranges
        for worksheet in self.workbook.worksheets():
            if worksheet.is_chartsheet:
                self.chartsheet_count += 1
            else:
                self.worksheet_count += 1

    def _create_package(self):
        if False:
            return 10
        self._write_content_types_file()
        self._write_root_rels_file()
        self._write_workbook_rels_file()
        self._write_worksheet_files()
        self._write_chartsheet_files()
        self._write_workbook_file()
        self._write_chart_files()
        self._write_drawing_files()
        self._write_vml_files()
        self._write_comment_files()
        self._write_table_files()
        self._write_shared_strings_file()
        self._write_styles_file()
        self._write_custom_file()
        self._write_theme_file()
        self._write_worksheet_rels_files()
        self._write_chartsheet_rels_files()
        self._write_drawing_rels_files()
        self._add_image_files()
        self._add_vba_project()
        self._add_vba_project_signature()
        self._write_vba_project_rels_file()
        self._write_core_file()
        self._write_app_file()
        self._write_metadata_file()
        return self.filenames

    def _filename(self, xml_filename):
        if False:
            for i in range(10):
                print('nop')
        if self.in_memory:
            os_filename = StringIO()
        else:
            (fd, os_filename) = tempfile.mkstemp(dir=self.tmpdir)
            os.close(fd)
        self.filenames.append((os_filename, xml_filename, False))
        return os_filename

    def _write_workbook_file(self):
        if False:
            for i in range(10):
                print('nop')
        workbook = self.workbook
        workbook._set_xml_writer(self._filename('xl/workbook.xml'))
        workbook._assemble_xml_file()

    def _write_worksheet_files(self):
        if False:
            i = 10
            return i + 15
        index = 1
        for worksheet in self.workbook.worksheets():
            if worksheet.is_chartsheet:
                continue
            if worksheet.constant_memory:
                worksheet._opt_reopen()
                worksheet._write_single_row()
            worksheet._set_xml_writer(self._filename('xl/worksheets/sheet' + str(index) + '.xml'))
            worksheet._assemble_xml_file()
            index += 1

    def _write_chartsheet_files(self):
        if False:
            while True:
                i = 10
        index = 1
        for worksheet in self.workbook.worksheets():
            if not worksheet.is_chartsheet:
                continue
            worksheet._set_xml_writer(self._filename('xl/chartsheets/sheet' + str(index) + '.xml'))
            worksheet._assemble_xml_file()
            index += 1

    def _write_chart_files(self):
        if False:
            while True:
                i = 10
        if not self.workbook.charts:
            return
        index = 1
        for chart in self.workbook.charts:
            if not chart.series:
                raise EmptyChartSeries('Chart%d must contain at least one data series. See chart.add_series().' % index)
            chart._set_xml_writer(self._filename('xl/charts/chart' + str(index) + '.xml'))
            chart._assemble_xml_file()
            index += 1

    def _write_drawing_files(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.drawing_count:
            return
        index = 1
        for drawing in self.workbook.drawings:
            drawing._set_xml_writer(self._filename('xl/drawings/drawing' + str(index) + '.xml'))
            drawing._assemble_xml_file()
            index += 1

    def _write_vml_files(self):
        if False:
            return 10
        index = 1
        for worksheet in self.workbook.worksheets():
            if not worksheet.has_vml and (not worksheet.has_header_vml):
                continue
            if worksheet.has_vml:
                vml = Vml()
                vml._set_xml_writer(self._filename('xl/drawings/vmlDrawing' + str(index) + '.vml'))
                vml._assemble_xml_file(worksheet.vml_data_id, worksheet.vml_shape_id, worksheet.comments_list, worksheet.buttons_list)
                index += 1
            if worksheet.has_header_vml:
                vml = Vml()
                vml._set_xml_writer(self._filename('xl/drawings/vmlDrawing' + str(index) + '.vml'))
                vml._assemble_xml_file(worksheet.vml_header_id, worksheet.vml_header_id * 1024, None, None, worksheet.header_images_list)
                self._write_vml_drawing_rels_file(worksheet, index)
                index += 1

    def _write_comment_files(self):
        if False:
            i = 10
            return i + 15
        index = 1
        for worksheet in self.workbook.worksheets():
            if not worksheet.has_comments:
                continue
            comment = Comments()
            comment._set_xml_writer(self._filename('xl/comments' + str(index) + '.xml'))
            comment._assemble_xml_file(worksheet.comments_list)
            index += 1

    def _write_shared_strings_file(self):
        if False:
            for i in range(10):
                print('nop')
        sst = SharedStrings()
        sst.string_table = self.workbook.str_table
        if not self.workbook.str_table.count:
            return
        sst._set_xml_writer(self._filename('xl/sharedStrings.xml'))
        sst._assemble_xml_file()

    def _write_app_file(self):
        if False:
            while True:
                i = 10
        properties = self.workbook.doc_properties
        app = App()
        worksheet_count = 0
        for worksheet in self.workbook.worksheets():
            if worksheet.is_chartsheet:
                continue
            if worksheet.hidden != 2:
                app._add_part_name(worksheet.name)
                worksheet_count += 1
        app._add_heading_pair(['Worksheets', worksheet_count])
        for worksheet in self.workbook.worksheets():
            if not worksheet.is_chartsheet:
                continue
            app._add_part_name(worksheet.name)
        app._add_heading_pair(['Charts', self.chartsheet_count])
        if self.named_ranges:
            app._add_heading_pair(['Named Ranges', len(self.named_ranges)])
        for named_range in self.named_ranges:
            app._add_part_name(named_range)
        app._set_properties(properties)
        app.doc_security = self.workbook.read_only
        app._set_xml_writer(self._filename('docProps/app.xml'))
        app._assemble_xml_file()

    def _write_core_file(self):
        if False:
            while True:
                i = 10
        properties = self.workbook.doc_properties
        core = Core()
        core._set_properties(properties)
        core._set_xml_writer(self._filename('docProps/core.xml'))
        core._assemble_xml_file()

    def _write_metadata_file(self):
        if False:
            return 10
        if not self.workbook.has_metadata:
            return
        metadata = Metadata()
        metadata._set_xml_writer(self._filename('xl/metadata.xml'))
        metadata._assemble_xml_file()

    def _write_custom_file(self):
        if False:
            while True:
                i = 10
        properties = self.workbook.custom_properties
        custom = Custom()
        if not len(properties):
            return
        custom._set_properties(properties)
        custom._set_xml_writer(self._filename('docProps/custom.xml'))
        custom._assemble_xml_file()

    def _write_content_types_file(self):
        if False:
            i = 10
            return i + 15
        content = ContentTypes()
        content._add_image_types(self.workbook.image_types)
        self._get_table_count()
        worksheet_index = 1
        chartsheet_index = 1
        for worksheet in self.workbook.worksheets():
            if worksheet.is_chartsheet:
                content._add_chartsheet_name('sheet' + str(chartsheet_index))
                chartsheet_index += 1
            else:
                content._add_worksheet_name('sheet' + str(worksheet_index))
                worksheet_index += 1
        for i in range(1, self.chart_count + 1):
            content._add_chart_name('chart' + str(i))
        for i in range(1, self.drawing_count + 1):
            content._add_drawing_name('drawing' + str(i))
        if self.num_vml_files:
            content._add_vml_name()
        for i in range(1, self.table_count + 1):
            content._add_table_name('table' + str(i))
        for i in range(1, self.num_comment_files + 1):
            content._add_comment_name('comments' + str(i))
        if self.workbook.str_table.count:
            content._add_shared_strings()
        if self.workbook.vba_project:
            content._add_vba_project()
            if self.workbook.vba_project_signature:
                content._add_vba_project_signature()
        if self.workbook.custom_properties:
            content._add_custom_properties()
        if self.workbook.has_metadata:
            content._add_metadata()
        content._set_xml_writer(self._filename('[Content_Types].xml'))
        content._assemble_xml_file()

    def _write_styles_file(self):
        if False:
            i = 10
            return i + 15
        xf_formats = self.workbook.xf_formats
        palette = self.workbook.palette
        font_count = self.workbook.font_count
        num_formats = self.workbook.num_formats
        border_count = self.workbook.border_count
        fill_count = self.workbook.fill_count
        custom_colors = self.workbook.custom_colors
        dxf_formats = self.workbook.dxf_formats
        has_comments = self.workbook.has_comments
        styles = Styles()
        styles._set_style_properties([xf_formats, palette, font_count, num_formats, border_count, fill_count, custom_colors, dxf_formats, has_comments])
        styles._set_xml_writer(self._filename('xl/styles.xml'))
        styles._assemble_xml_file()

    def _write_theme_file(self):
        if False:
            while True:
                i = 10
        theme = Theme()
        theme._set_xml_writer(self._filename('xl/theme/theme1.xml'))
        theme._assemble_xml_file()

    def _write_table_files(self):
        if False:
            return 10
        index = 1
        for worksheet in self.workbook.worksheets():
            table_props = worksheet.tables
            if not table_props:
                continue
            for table_props in table_props:
                table = Table()
                table._set_xml_writer(self._filename('xl/tables/table' + str(index) + '.xml'))
                table._set_properties(table_props)
                table._assemble_xml_file()
                index += 1

    def _get_table_count(self):
        if False:
            i = 10
            return i + 15
        for worksheet in self.workbook.worksheets():
            for _ in worksheet.tables:
                self.table_count += 1

    def _write_root_rels_file(self):
        if False:
            i = 10
            return i + 15
        rels = Relationships()
        rels._add_document_relationship('/officeDocument', 'xl/workbook.xml')
        rels._add_package_relationship('/metadata/core-properties', 'docProps/core.xml')
        rels._add_document_relationship('/extended-properties', 'docProps/app.xml')
        if self.workbook.custom_properties:
            rels._add_document_relationship('/custom-properties', 'docProps/custom.xml')
        rels._set_xml_writer(self._filename('_rels/.rels'))
        rels._assemble_xml_file()

    def _write_workbook_rels_file(self):
        if False:
            for i in range(10):
                print('nop')
        rels = Relationships()
        worksheet_index = 1
        chartsheet_index = 1
        for worksheet in self.workbook.worksheets():
            if worksheet.is_chartsheet:
                rels._add_document_relationship('/chartsheet', 'chartsheets/sheet' + str(chartsheet_index) + '.xml')
                chartsheet_index += 1
            else:
                rels._add_document_relationship('/worksheet', 'worksheets/sheet' + str(worksheet_index) + '.xml')
                worksheet_index += 1
        rels._add_document_relationship('/theme', 'theme/theme1.xml')
        rels._add_document_relationship('/styles', 'styles.xml')
        if self.workbook.str_table.count:
            rels._add_document_relationship('/sharedStrings', 'sharedStrings.xml')
        if self.workbook.vba_project:
            rels._add_ms_package_relationship('/vbaProject', 'vbaProject.bin')
        if self.workbook.has_metadata:
            rels._add_document_relationship('/sheetMetadata', 'metadata.xml')
        rels._set_xml_writer(self._filename('xl/_rels/workbook.xml.rels'))
        rels._assemble_xml_file()

    def _write_worksheet_rels_files(self):
        if False:
            print('Hello World!')
        index = 0
        for worksheet in self.workbook.worksheets():
            if worksheet.is_chartsheet:
                continue
            index += 1
            external_links = worksheet.external_hyper_links + worksheet.external_drawing_links + worksheet.external_vml_links + worksheet.external_background_links + worksheet.external_table_links + worksheet.external_comment_links
            if not external_links:
                continue
            rels = Relationships()
            for link_data in external_links:
                rels._add_document_relationship(*link_data)
            rels._set_xml_writer(self._filename('xl/worksheets/_rels/sheet' + str(index) + '.xml.rels'))
            rels._assemble_xml_file()

    def _write_chartsheet_rels_files(self):
        if False:
            for i in range(10):
                print('nop')
        index = 0
        for worksheet in self.workbook.worksheets():
            if not worksheet.is_chartsheet:
                continue
            index += 1
            external_links = worksheet.external_drawing_links + worksheet.external_vml_links
            if not external_links:
                continue
            rels = Relationships()
            for link_data in external_links:
                rels._add_document_relationship(*link_data)
            rels._set_xml_writer(self._filename('xl/chartsheets/_rels/sheet' + str(index) + '.xml.rels'))
            rels._assemble_xml_file()

    def _write_drawing_rels_files(self):
        if False:
            return 10
        index = 0
        for worksheet in self.workbook.worksheets():
            if worksheet.drawing:
                index += 1
            if not worksheet.drawing_links:
                continue
            rels = Relationships()
            for drawing_data in worksheet.drawing_links:
                rels._add_document_relationship(*drawing_data)
            rels._set_xml_writer(self._filename('xl/drawings/_rels/drawing' + str(index) + '.xml.rels'))
            rels._assemble_xml_file()

    def _write_vml_drawing_rels_file(self, worksheet, index):
        if False:
            i = 10
            return i + 15
        rels = Relationships()
        for drawing_data in worksheet.vml_drawing_links:
            rels._add_document_relationship(*drawing_data)
        rels._set_xml_writer(self._filename('xl/drawings/_rels/vmlDrawing' + str(index) + '.vml.rels'))
        rels._assemble_xml_file()

    def _write_vba_project_rels_file(self):
        if False:
            print('Hello World!')
        vba_project_signature = self.workbook.vba_project_signature
        if not vba_project_signature:
            return
        rels = Relationships()
        rels._add_ms_package_relationship('/vbaProjectSignature', 'vbaProjectSignature.bin')
        rels._set_xml_writer(self._filename('xl/_rels/vbaProject.bin.rels'))
        rels._assemble_xml_file()

    def _add_image_files(self):
        if False:
            return 10
        workbook = self.workbook
        index = 1
        for image in workbook.images:
            filename = image[0]
            ext = '.' + image[1]
            image_data = image[2]
            xml_image_name = 'xl/media/image' + str(index) + ext
            if not self.in_memory:
                os_filename = self._filename(xml_image_name)
                if image_data:
                    os_file = open(os_filename, mode='wb')
                    os_file.write(image_data.getvalue())
                    os_file.close()
                else:
                    copy(filename, os_filename)
                    try:
                        os.chmod(os_filename, os.stat(os_filename).st_mode | stat.S_IWRITE)
                    except OSError:
                        pass
            else:
                if image_data:
                    os_filename = image_data
                else:
                    image_file = open(filename, mode='rb')
                    image_data = image_file.read()
                    os_filename = BytesIO(image_data)
                    image_file.close()
                self.filenames.append((os_filename, xml_image_name, True))
            index += 1

    def _add_vba_project_signature(self):
        if False:
            for i in range(10):
                print('nop')
        vba_project_signature = self.workbook.vba_project_signature
        vba_project_signature_is_stream = self.workbook.vba_project_signature_is_stream
        if not vba_project_signature:
            return
        xml_vba_signature_name = 'xl/vbaProjectSignature.bin'
        if not self.in_memory:
            os_filename = self._filename(xml_vba_signature_name)
            if vba_project_signature_is_stream:
                os_file = open(os_filename, mode='wb')
                os_file.write(vba_project_signature.getvalue())
                os_file.close()
            else:
                copy(vba_project_signature, os_filename)
        else:
            if vba_project_signature_is_stream:
                os_filename = vba_project_signature
            else:
                vba_file = open(vba_project_signature, mode='rb')
                vba_data = vba_file.read()
                os_filename = BytesIO(vba_data)
                vba_file.close()
            self.filenames.append((os_filename, xml_vba_signature_name, True))

    def _add_vba_project(self):
        if False:
            while True:
                i = 10
        vba_project = self.workbook.vba_project
        vba_project_is_stream = self.workbook.vba_project_is_stream
        if not vba_project:
            return
        xml_vba_name = 'xl/vbaProject.bin'
        if not self.in_memory:
            os_filename = self._filename(xml_vba_name)
            if vba_project_is_stream:
                os_file = open(os_filename, mode='wb')
                os_file.write(vba_project.getvalue())
                os_file.close()
            else:
                copy(vba_project, os_filename)
        else:
            if vba_project_is_stream:
                os_filename = vba_project
            else:
                vba_file = open(vba_project, mode='rb')
                vba_data = vba_file.read()
                os_filename = BytesIO(vba_data)
                vba_file.close()
            self.filenames.append((os_filename, xml_vba_name, True))