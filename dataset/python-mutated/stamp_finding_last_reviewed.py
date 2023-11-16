from auditlog.models import LogEntry
from django.contrib.contenttypes.models import ContentType
from django.core.management.base import BaseCommand
from pytz import timezone
from dojo.models import Finding
from dojo.utils import get_system_setting
locale = timezone(get_system_setting('time_zone'))
'\nAuthors: Jay Paz\nNew fields last_reviewed, last_reviewed_by, mitigated_by have been added to the Finding model\nThis script will update all findings with a last_reviewed date of the most current date from:\n1.  Finding Date if no other evidence of activity is found\n2.  Last note added date if a note is found\n3.  Mitigation Date if finding is mitigated\n4.  Last action_log entry date if Finding has been updated\n\nIt will update the last_reviewed_by with the current reporter.\n\nIf mitigated it will update the mitigated_by with last_reviewed_by or current reporter if last_reviewed_by is None\n'

class Command(BaseCommand):
    help = 'A new field last_reviewed has been added to the Finding model \nThis script will update all findings with a last_reviewed date of the most current date from: \n1.  Finding Date if no other evidence of activity is found \n2.  Last note added date if a note is found \n3.  Mitigation Date if finding is mitigated \n4.  Last action_log entry date if Finding has been updated \n'

    def handle(self, *args, **options):
        if False:
            return 10
        findings = Finding.objects.all().order_by('id')
        for finding in findings:
            save = False
            if not finding.last_reviewed:
                date_discovered = finding.date
                last_note_date = finding.date
                if finding.notes.all():
                    last_note_date = finding.notes.order_by('-date')[0].date.date()
                mitigation_date = finding.date
                if finding.mitigated:
                    mitigation_date = finding.mitigated.date()
                last_action_date = finding.date
                try:
                    ct = ContentType.objects.get_for_id(ContentType.objects.get_for_model(finding).id)
                    obj = ct.get_object_for_this_type(pk=finding.id)
                    log_entries = LogEntry.objects.filter(content_type=ct, object_pk=obj.id).order_by('-timestamp')
                    if log_entries:
                        last_action_date = log_entries[0].timestamp.date()
                except KeyError:
                    pass
                finding.last_reviewed = max([date_discovered, last_note_date, mitigation_date, last_action_date])
                save = True
            if not finding.last_reviewed_by:
                finding.last_reviewed_by = finding.reporter
                save = True
            if finding.mitigated:
                if not finding.mitigated_by:
                    finding.mitigated_by = finding.last_reviewed_by if finding.last_reviewed_by else finding.reporter
                    save = True
            if save:
                finding.save()