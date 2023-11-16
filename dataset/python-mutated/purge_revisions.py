from django.conf import settings
from django.core.management.base import BaseCommand
from django.db.models import Q
from django.db.models.deletion import ProtectedError
from django.utils import timezone
from wagtail.models import Revision, WorkflowState

class Command(BaseCommand):
    help = 'Delete revisions which are not the latest revision, published or scheduled to be published, or in moderation'

    def add_arguments(self, parser):
        if False:
            i = 10
            return i + 15
        parser.add_argument('--days', type=int, help='Only delete revisions older than this number of days')
        parser.add_argument('--pages', action='store_true', help='Only delete revisions of page models')
        parser.add_argument('--non-pages', action='store_true', help='Only delete revisions of non-page models')

    def handle(self, *args, **options):
        if False:
            return 10
        days = options.get('days')
        pages = options.get('pages')
        non_pages = options.get('non_pages')
        (revisions_deleted, protected_error_count) = purge_revisions(days=days, pages=pages, non_pages=non_pages)
        if revisions_deleted:
            self.stdout.write(self.style.SUCCESS('Successfully deleted %s revisions' % revisions_deleted))
            self.stdout.write(self.style.SUCCESS('Ignored %s revisions because one or more protected relations exist that prevent deletion.' % protected_error_count))
        else:
            self.stdout.write('No revisions deleted')

def purge_revisions(days=None, pages=True, non_pages=True):
    if False:
        for i in range(10):
            print('nop')
    if pages == non_pages:
        objects = Revision.objects.all()
    elif pages:
        objects = Revision.objects.page_revisions()
    elif non_pages:
        objects = Revision.objects.not_page_revisions()
    purgeable_revisions = objects.exclude(approved_go_live_at__isnull=False)
    if getattr(settings, 'WAGTAIL_WORKFLOW_ENABLED', True):
        purgeable_revisions = purgeable_revisions.exclude(Q(task_states__workflow_state__status=WorkflowState.STATUS_IN_PROGRESS) | Q(task_states__workflow_state__status=WorkflowState.STATUS_NEEDS_CHANGES))
    if days:
        purgeable_until = timezone.now() - timezone.timedelta(days=days)
        purgeable_revisions = purgeable_revisions.filter(created_at__lt=purgeable_until)
    deleted_revisions_count = 0
    protected_error_count = 0
    for revision in purgeable_revisions.iterator():
        if not revision.is_latest_revision():
            try:
                revision.delete()
                deleted_revisions_count += 1
            except ProtectedError:
                protected_error_count += 1
    return (deleted_revisions_count, protected_error_count)