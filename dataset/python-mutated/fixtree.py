import functools
import operator
from django.core.management.base import BaseCommand
from django.db import models
from django.db.models import Q
from wagtail.models import Collection, Page

class Command(BaseCommand):
    help = 'Checks for data integrity errors on the page tree, and fixes them where possible.'
    stealth_options = ('delete_orphans',)

    def add_arguments(self, parser):
        if False:
            print('Hello World!')
        parser.add_argument('--noinput', action='store_false', dest='interactive', default=True, help='If provided, any fixes requiring user interaction will be skipped.')
        parser.add_argument('--full', action='store_true', dest='full', default=False, help='If provided, uses a more thorough but slower method that also fixes path ordering issues.')

    def numberlist_to_string(self, numberlist):
        if False:
            print('Hello World!')
        return '[' + ', '.join(map(str, numberlist)) + ']'

    def handle(self, **options):
        if False:
            for i in range(10):
                print('nop')
        any_page_problems_fixed = False
        for page in Page.objects.all():
            try:
                page.specific
            except page.specific_class.DoesNotExist:
                self.stdout.write('Page %d (%s) is missing a subclass record; deleting.' % (page.id, page.title))
                any_page_problems_fixed = True
                page.delete()
        self.handle_model(Page, 'page', 'pages', any_page_problems_fixed, options)
        self.handle_model(Collection, 'collection', 'collections', False, options)

    def handle_model(self, model, model_name, model_name_plural, any_problems_fixed, options):
        if False:
            return 10
        fix_paths = options.get('full', False)
        self.stdout.write('Checking %s tree for problems...' % model_name)
        (bad_alpha, bad_path, orphans, bad_depth, bad_numchild) = model.find_problems()
        if bad_depth:
            self.stdout.write('Incorrect depth value found for %s: %s' % (model_name_plural, self.numberlist_to_string(bad_depth)))
        if bad_numchild:
            self.stdout.write('Incorrect numchild value found for %s: %s' % (model_name_plural, self.numberlist_to_string(bad_numchild)))
        if orphans:
            orphan_paths = model.objects.filter(id__in=orphans).values_list('path', flat=True)
            filter_conditions = []
            for path in orphan_paths:
                filter_conditions.append(Q(path__startswith=path))
            final_filter = functools.reduce(operator.or_, filter_conditions)
            nodes_to_delete = models.query.QuerySet(model).filter(final_filter)
            self.stdout.write('Orphaned %s found:' % model_name_plural)
            for node in nodes_to_delete:
                self.stdout.write('ID %d: %s' % (node.id, node))
            self.stdout.write('')
            if options.get('interactive', True):
                yes_or_no = input('Delete these %s? [y/N] ' % model_name_plural)
                delete_orphans = yes_or_no.lower().startswith('y')
                self.stdout.write('')
            else:
                delete_orphans = options.get('delete_orphans', False)
            if delete_orphans:
                deletion_count = len(nodes_to_delete)
                nodes_to_delete.delete()
                self.stdout.write('%d orphaned %s deleted.' % (deletion_count, model_name_plural if deletion_count != 1 else model_name))
                any_problems_fixed = True
        if bad_depth or bad_numchild or fix_paths:
            model.fix_tree(destructive=False, fix_paths=fix_paths)
            any_problems_fixed = True
        if any_problems_fixed:
            (bad_alpha, bad_path, orphans, bad_depth, bad_numchild) = model.find_problems()
        if any((bad_alpha, bad_path, orphans, bad_depth, bad_numchild)):
            self.stdout.write('Remaining problems (cannot fix automatically):')
            if bad_alpha:
                self.stdout.write('Invalid characters found in path for %s: %s' % (model_name_plural, self.numberlist_to_string(bad_alpha)))
            if bad_path:
                self.stdout.write('Invalid path length found for %s: %s' % (model_name_plural, self.numberlist_to_string(bad_path)))
            if orphans:
                self.stdout.write('Orphaned %s found: %s' % (model_name_plural, self.numberlist_to_string(orphans)))
            if bad_depth:
                self.stdout.write('Incorrect depth value found for %s: %s' % (model_name_plural, self.numberlist_to_string(bad_depth)))
            if bad_numchild:
                self.stdout.write('Incorrect numchild value found for %s: %s' % (model_name_plural, self.numberlist_to_string(bad_numchild)))
        elif any_problems_fixed:
            self.stdout.write('All problems fixed.\n\n')
        else:
            self.stdout.write('No problems found.\n\n')