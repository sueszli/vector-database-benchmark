from django.core.management.base import BaseCommand
from ...search_tasks import set_order_search_document_values, set_product_search_document_values, set_user_search_document_values

class Command(BaseCommand):
    help = 'Populate search indexes.'

    def handle(self, *args, **options):
        if False:
            for i in range(10):
                print('nop')
        self.stdout.write('Updating products')
        set_product_search_document_values.delay()
        self.stdout.write('Updating orders')
        set_order_search_document_values.delay()
        self.stdout.write('Updating users')
        set_user_search_document_values.delay()