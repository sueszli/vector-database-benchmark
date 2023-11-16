"""Label printing plugin which supports printing multiple labels on a single page"""
import logging
import math
from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile
from django.http import JsonResponse
from django.utils.translation import gettext_lazy as _
import weasyprint
from rest_framework import serializers
import report.helpers
from label.models import LabelOutput, LabelTemplate
from plugin import InvenTreePlugin
from plugin.mixins import LabelPrintingMixin, SettingsMixin
logger = logging.getLogger('inventree')

class LabelPrintingOptionsSerializer(serializers.Serializer):
    """Custom printing options for the label sheet plugin"""
    page_size = serializers.ChoiceField(choices=report.helpers.report_page_size_options(), default='A4', label=_('Page Size'), help_text=_('Page size for the label sheet'))
    border = serializers.BooleanField(default=False, label=_('Border'), help_text=_('Print a border around each label'))
    landscape = serializers.BooleanField(default=False, label=_('Landscape'), help_text=_('Print the label sheet in landscape mode'))

class InvenTreeLabelSheetPlugin(LabelPrintingMixin, SettingsMixin, InvenTreePlugin):
    """Builtin plugin for label printing.

    This plugin arrays multiple labels onto a single larger sheet,
    and returns the resulting PDF file.
    """
    NAME = 'InvenTreeLabelSheet'
    TITLE = _('InvenTree Label Sheet Printer')
    DESCRIPTION = _('Arrays multiple labels onto a single sheet')
    VERSION = '1.0.0'
    AUTHOR = _('InvenTree contributors')
    BLOCKING_PRINT = True
    SETTINGS = {}
    PrintingOptionsSerializer = LabelPrintingOptionsSerializer

    def print_labels(self, label: LabelTemplate, items: list, request, **kwargs):
        if False:
            print('Hello World!')
        'Handle printing of the provided labels'
        printing_options = kwargs['printing_options']
        page_size_code = printing_options.get('page_size', 'A4')
        landscape = printing_options.get('landscape', False)
        border = printing_options.get('border', False)
        page_size = report.helpers.page_size(page_size_code)
        (page_width, page_height) = page_size
        if landscape:
            (page_width, page_height) = (page_height, page_width)
        n_cols = math.floor(page_width / label.width)
        n_rows = math.floor(page_height / label.height)
        n_cells = n_cols * n_rows
        if n_cells == 0:
            raise ValidationError(_('Label is too large for page size'))
        n_labels = len(items)
        document_data = {'border': border, 'landscape': landscape, 'page_width': page_width, 'page_height': page_height, 'label_width': label.width, 'label_height': label.height, 'n_labels': n_labels, 'n_pages': math.ceil(n_labels / n_cells), 'n_cols': n_cols, 'n_rows': n_rows}
        pages = []
        idx = 0
        while idx < n_labels:
            if (page := self.print_page(label, items[idx:idx + n_cells], request, **document_data)):
                pages.append(page)
            idx += n_cells
        if len(pages) == 0:
            raise ValidationError(_('No labels were generated'))
        html_data = self.wrap_pages(pages, **document_data)
        html = weasyprint.HTML(string=html_data)
        document = html.render().write_pdf()
        output_file = ContentFile(document, 'labels.pdf')
        output = LabelOutput.objects.create(label=output_file, user=request.user)
        return JsonResponse({'file': output.label.url, 'success': True, 'message': f'{len(items)} labels generated'})

    def print_page(self, label: LabelTemplate, items: list, request, **kwargs):
        if False:
            return 10
        'Generate a single page of labels:\n\n        For a single page, generate a simple table grid of labels.\n        Styling of the table is handled by the higher level label template\n\n        Arguments:\n            label: The LabelTemplate object to use for printing\n            items: The list of database items to print (e.g. StockItem instances)\n            request: The HTTP request object which triggered this print job\n\n        Kwargs:\n            n_cols: Number of columns\n            n_rows: Number of rows\n        '
        n_cols = kwargs['n_cols']
        n_rows = kwargs['n_rows']
        html = "<table class='label-sheet-table'>"
        for row in range(n_rows):
            html += "<tr class='label-sheet-row'>"
            for col in range(n_cols):
                html += f"<td class='label-sheet-cell label-sheet-row-{row} label-sheet-col-{col}'>"
                idx = row * n_cols + col
                if idx < len(items):
                    try:
                        cell = label.render_as_string(request, target_object=items[idx], insert_page_style=False)
                        html += cell
                    except Exception as exc:
                        logger.exception('Error rendering label: %s', str(exc))
                        html += "\n                        <div class='label-sheet-cell-error'></div>\n                        "
                html += '</td>'
            html += '</tr>'
        html += '</table>'
        return html

    def wrap_pages(self, pages, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Wrap the generated pages into a single document'
        border = kwargs['border']
        page_width = kwargs['page_width']
        page_height = kwargs['page_height']
        label_width = kwargs['label_width']
        label_height = kwargs['label_height']
        n_rows = kwargs['n_rows']
        n_cols = kwargs['n_cols']
        inner = ''.join(pages)
        cell_styles = []
        for row in range(n_rows):
            cell_styles.append(f'\n            .label-sheet-row-{row} {{\n                top: {row * label_height}mm;\n            }}\n            ')
        for col in range(n_cols):
            cell_styles.append(f'\n            .label-sheet-col-{col} {{\n                left: {col * label_width}mm;\n            }}\n            ')
        cell_styles = '\n'.join(cell_styles)
        return f"\n        <head>\n            <style>\n                @page {{\n                    size: {page_width}mm {page_height}mm;\n                    margin: 0mm;\n                    padding: 0mm;\n                }}\n\n                .label-sheet-table {{\n                    page-break-after: always;\n                    table-layout: fixed;\n                    width: {page_width}mm;\n                    border-spacing: 0mm 0mm;\n                }}\n\n                .label-sheet-cell-error {{\n                    background-color: #F00;\n                }}\n\n                .label-sheet-cell {{\n                    border: {('1px solid #000;' if border else '0mm;')}\n                    width: {label_width}mm;\n                    height: {label_height}mm;\n                    padding: 0mm;\n                    position: absolute;\n                }}\n\n                {cell_styles}\n\n                body {{\n                    margin: 0mm !important;\n                }}\n            </style>\n        </head>\n        <body>\n            {inner}\n        </body>\n        </html>\n        "