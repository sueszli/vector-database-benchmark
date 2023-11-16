from awxkit.api.resources import resources
from awxkit.utils import random_title, update_payload
from awxkit.api.mixins import HasStatus
from . import base
from . import page

class UnifiedJobTemplate(HasStatus, base.Base):
    """Base class for unified job template pages (e.g. project, inventory_source,
    and job_template).
    """
    optional_schedule_fields = ('extra_data', 'diff_mode', 'limit', 'job_tags', 'skip_tags', 'job_type', 'verbosity', 'inventory', 'forks', 'timeout', 'job_slice_count', 'execution_environment')

    def __str__(self):
        if False:
            i = 10
            return i + 15
        items = ['id', 'name', 'status', 'source', 'last_update_failed', 'last_updated', 'result_traceback', 'job_explanation', 'job_args']
        info = []
        for item in [x for x in items if hasattr(self, x)]:
            info.append('{0}:{1}'.format(item, getattr(self, item)))
        output = '<{0.__class__.__name__} {1}>'.format(self, ', '.join(info))
        return output.replace('%', '%%')

    def add_schedule(self, name='', description='', enabled=True, rrule=None, **kwargs):
        if False:
            return 10
        if rrule is None:
            rrule = 'DTSTART:30180101T000000Z RRULE:FREQ=YEARLY;INTERVAL=1'
        payload = dict(name=name or '{0} Schedule {1}'.format(self.name, random_title()), description=description or random_title(10), enabled=enabled, rrule=str(rrule))
        update_payload(payload, self.optional_schedule_fields, kwargs)
        schedule = self.related.schedules.post(payload)
        if not hasattr(self, '_schedules_store'):
            self._schedules_store = set()
        if schedule not in self._schedules_store:
            self._schedules_store.add(schedule)
        return schedule

    def silent_delete(self):
        if False:
            i = 10
            return i + 15
        if hasattr(self, '_schedules_store'):
            for schedule in self._schedules_store:
                schedule.silent_delete()
        return super(UnifiedJobTemplate, self).silent_delete()

    @property
    def is_successful(self):
        if False:
            return 10
        "An unified_job_template is considered successful when:\n        1) status == 'successful'\n        2) not last_update_failed\n        3) last_updated\n        "
        return super(UnifiedJobTemplate, self).is_successful and (not self.last_update_failed) and (self.last_updated is not None)
page.register_page(resources.unified_job_template, UnifiedJobTemplate)

class UnifiedJobTemplates(page.PageList, UnifiedJobTemplate):
    pass
page.register_page(resources.unified_job_templates, UnifiedJobTemplates)