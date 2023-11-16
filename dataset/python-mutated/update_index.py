import collections
from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import transaction
from wagtail.search.backends import get_search_backend
from wagtail.search.index import get_indexed_models
DEFAULT_CHUNK_SIZE = 1000

def group_models_by_index(backend, models):
    if False:
        return 10
    '\n    This takes a search backend and a list of models. By calling the\n    get_index_for_model method on the search backend, it groups the models into\n    the indices that they will be indexed into.\n\n    It returns an ordered mapping of indices to lists of models within each\n    index.\n\n    For example, Elasticsearch 2 requires all page models to be together, but\n    separate from other content types (eg, images and documents) to prevent\n    field mapping collisions:\n\n    >>> group_models_by_index(elasticsearch2_backend, [\n    ...     wagtailcore.Page,\n    ...     myapp.HomePage,\n    ...     myapp.StandardPage,\n    ...     wagtailimages.Image\n    ... ])\n    {\n        <Index wagtailcore_page>: [wagtailcore.Page, myapp.HomePage, myapp.StandardPage],\n        <Index wagtailimages_image>: [wagtailimages.Image],\n    }\n    '
    indices = {}
    models_by_index = collections.OrderedDict()
    for model in models:
        index = backend.get_index_for_model(model)
        if index:
            indices.setdefault(index.name, index)
            models_by_index.setdefault(index.name, [])
            models_by_index[index.name].append(model)
    return collections.OrderedDict([(indices[index_name], index_models) for (index_name, index_models) in models_by_index.items()])

class Command(BaseCommand):

    def write(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Helper function that respects verbosity when printing.'
        if self.verbosity > 0:
            self.stdout.write(*args, **kwargs)

    def update_backend(self, backend_name, schema_only=False, chunk_size=DEFAULT_CHUNK_SIZE):
        if False:
            while True:
                i = 10
        self.write('Updating backend: ' + backend_name)
        backend = get_search_backend(backend_name)
        if not backend.rebuilder_class:
            self.write("Backend '%s' doesn't require rebuilding" % backend_name)
            return
        models_grouped_by_index = group_models_by_index(backend, get_indexed_models()).items()
        if not models_grouped_by_index:
            self.write(backend_name + ': No indices to rebuild')
        for (index, models) in models_grouped_by_index:
            self.write(backend_name + ': Rebuilding index %s' % index.name)
            rebuilder = backend.rebuilder_class(index)
            index = rebuilder.start()
            for model in models:
                index.add_model(model)
            object_count = 0
            if not schema_only:
                for model in models:
                    self.write('{}: {}.{} '.format(backend_name, model._meta.app_label, model.__name__).ljust(35), ending='')
                    for chunk in self.print_iter_progress(self.queryset_chunks(model.get_indexed_objects().order_by('pk'), chunk_size)):
                        index.add_items(model, chunk)
                        object_count += len(chunk)
                    self.print_newline()
            rebuilder.finish()
            self.write(backend_name + ': indexed %d objects' % object_count)
            self.print_newline()

    def add_arguments(self, parser):
        if False:
            for i in range(10):
                print('nop')
        parser.add_argument('--backend', action='store', dest='backend_name', default=None, help='Specify a backend to update')
        parser.add_argument('--schema-only', action='store_true', dest='schema_only', default=False, help='Prevents loading any data into the index')
        parser.add_argument('--chunk_size', action='store', dest='chunk_size', default=DEFAULT_CHUNK_SIZE, type=int, help='Set number of records to be fetched at once for inserting into the index')

    def handle(self, **options):
        if False:
            i = 10
            return i + 15
        self.verbosity = options['verbosity']
        if options['backend_name']:
            backend_names = [options['backend_name']]
        elif hasattr(settings, 'WAGTAILSEARCH_BACKENDS'):
            backend_names = settings.WAGTAILSEARCH_BACKENDS.keys()
        else:
            backend_names = ['default']
        for backend_name in backend_names:
            self.update_backend(backend_name, schema_only=options.get('schema_only', False), chunk_size=options.get('chunk_size'))

    def print_newline(self):
        if False:
            for i in range(10):
                print('nop')
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
            for i in range(10):
                print('nop')
        '\n        Yield a queryset in chunks of at most ``chunk_size``. The chunk yielded\n        will be a list, not a queryset. Iterating over the chunks is done in a\n        transaction so that the order and count of items in the queryset\n        remains stable.\n        '
        i = 0
        while True:
            items = list(qs[i * chunk_size:][:chunk_size])
            if not items:
                break
            yield items
            i += 1