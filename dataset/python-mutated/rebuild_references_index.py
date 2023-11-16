from django.apps import apps
from django.core.management.base import BaseCommand
from django.db import transaction
from wagtail.models import ReferenceIndex
from wagtail.signal_handlers import disable_reference_index_auto_update
DEFAULT_CHUNK_SIZE = 1000

class Command(BaseCommand):

    def write(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Helper function that writes based on verbosity parameter\n\n        '
        if self.verbosity != 0:
            self.stdout.write(*args, **kwargs)

    def add_arguments(self, parser):
        if False:
            while True:
                i = 10
        parser.add_argument('--chunk_size', action='store', dest='chunk_size', default=DEFAULT_CHUNK_SIZE, type=int, help='Set number of records to be fetched at once for inserting into the index')

    def handle(self, **options):
        if False:
            return 10
        self.verbosity = options['verbosity']
        chunk_size = options.get('chunk_size')
        object_count = 0
        self.write('Rebuilding reference index')
        with transaction.atomic():
            with disable_reference_index_auto_update():
                ReferenceIndex.objects.all().delete()
            for model in apps.get_models():
                if not ReferenceIndex.is_indexed(model):
                    continue
                self.write(str(model))
                for chunk in self.print_iter_progress(self.queryset_chunks(model.objects.all().order_by('pk'), chunk_size)):
                    for instance in chunk:
                        ReferenceIndex.create_or_update_for_object(instance)
                    object_count += len(chunk)
                self.print_newline()
        self.write('Indexed %d objects' % object_count)
        self.print_newline()

    def print_newline(self):
        if False:
            i = 10
            return i + 15
        self.write('')

    def print_iter_progress(self, iterable):
        if False:
            return 10
        '\n        Print a progress meter while iterating over an iterable. Use it as part\n        of a ``for`` loop::\n\n            for item in self.print_iter_progress(big_long_list):\n                self.do_expensive_computation(item)\n\n        A ``.`` character is printed for every value in the iterable,\n        a space every 10 items, and a new line every 50 items.\n        '
        for (i, value) in enumerate(iterable, start=1):
            yield value
            self.write('.', ending='')
            if i % 40 == 0:
                self.print_newline()
                self.write(' ' * 35, ending='')
            elif i % 10 == 0:
                self.write(' ', ending='')
            self.stdout.flush()

    @transaction.atomic
    def queryset_chunks(self, qs, chunk_size=DEFAULT_CHUNK_SIZE):
        if False:
            print('Hello World!')
        '\n        Yield a queryset in chunks of at most ``chunk_size``. The chunk yielded\n        will be a list, not a queryset. Iterating over the chunks is done in a\n        transaction so that the order and count of items in the queryset\n        remains stable.\n        '
        i = 0
        while True:
            items = list(qs[i * chunk_size:][:chunk_size])
            if not items:
                break
            yield items
            i += 1