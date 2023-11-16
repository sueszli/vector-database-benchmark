from django.core.management.base import BaseCommand, CommandError
from sentry.mail import mail_adapter
from sentry.models.organization import Organization
from sentry.models.project import Project
from sentry.utils.email import get_email_addresses

def handle_project(project: Project, stream) -> None:
    if False:
        return 10
    '\n    For every user that should receive ISSUE_ALERT notifications for a given\n    project, write a map of usernames to email addresses to the given stream\n    one entry per line.\n    '
    stream.write('# Project: %s\n' % project)
    users = mail_adapter.get_sendable_user_objects(project)
    users_map = {user.id: user for user in users}
    emails = get_email_addresses(users_map.keys(), project)
    for (user_id, email) in emails.items():
        stream.write(f'{users_map[user_id].username}: {email}\n')

class Command(BaseCommand):
    help = 'Dump addresses that would get an email notification'

    def add_arguments(self, parser):
        if False:
            return 10
        parser.add_argument('--organization', action='store', type=int, dest='organization', default=0, help='')
        parser.add_argument('--project', action='store', type=int, dest='project', default=0, help='')

    def handle(self, *args, **options):
        if False:
            for i in range(10):
                print('nop')
        if not (options['project'] or options['organization']):
            raise CommandError('Must specify either a project or organization')
        if options['organization']:
            projects = list(Organization.objects.get(pk=options['organization']).project_set.all())
        else:
            projects = [Project.objects.get(pk=options['project'])]
        for project in projects:
            handle_project(project, self.stdout)
            self.stdout.write('\n')