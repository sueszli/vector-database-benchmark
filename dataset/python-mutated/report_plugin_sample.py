"""Sample plugin for extending reporting functionality"""
import random
from plugin import InvenTreePlugin
from plugin.mixins import ReportMixin
from report.models import PurchaseOrderReport

class SampleReportPlugin(ReportMixin, InvenTreePlugin):
    """Sample plugin which provides extra context data to a report"""
    NAME = 'Sample Report Plugin'
    SLUG = 'samplereport'
    TITLE = 'Sample Report Plugin'
    DESCRIPTION = 'A sample plugin which provides extra context data to a report'
    VERSION = '1.0'

    def some_custom_function(self):
        if False:
            for i in range(10):
                print('nop')
        'Some custom function which is not required for the plugin to function'
        return random.randint(0, 100)

    def add_report_context(self, report_instance, model_instance, request, context):
        if False:
            return 10
        'Add example content to the report instance'
        context['random_text'] = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=20))
        context['random_int'] = self.some_custom_function()
        context['is_purchase_order'] = isinstance(report_instance, PurchaseOrderReport)
        context['request_method'] = request.method