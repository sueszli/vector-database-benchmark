from django.core.management.base import BaseCommand
from dojo.tools.factory import PARSERS
from dojo.models import Test_Type, Tool_Type

class Command(BaseCommand):
    help = 'Initializes Test_Types'

    def handle(self, *args, **options):
        if False:
            print('Hello World!')
        for scan_type in PARSERS:
            Test_Type.objects.get_or_create(name=scan_type)
            parser = PARSERS[scan_type]
            if hasattr(parser, 'requires_tool_type'):
                tool_type = parser.requires_tool_type(scan_type)
                if tool_type:
                    Tool_Type.objects.get_or_create(name=tool_type)