from django.core.management.base import BaseCommand, CommandError

class Command(BaseCommand):
    help = 'Creates a sample event in Sentry (if applicable)'

    def add_arguments(self, parser):
        if False:
            print('Hello World!')
        (parser.add_argument('--project', dest='project', help='project ID or team-slug/project-slug'),)
        (parser.add_argument('--platform', dest='platform'),)

    def handle(self, **options):
        if False:
            i = 10
            return i + 15
        from django.conf import settings
        from sentry.models.project import Project
        from sentry.utils.samples import create_sample_event
        if not options['project']:
            project = Project.objects.get(id=settings.SENTRY_PROJECT)
        elif options['project'].isdigit():
            project = Project.objects.get(id=options['project'])
        elif '/' in options['project']:
            (t_slug, p_slug) = options['project'].split('/', 1)
            project = Project.objects.get(slug=p_slug, teams__slug=t_slug)
        else:
            raise CommandError('Project must be specified as team-slug/project-slug or a project id')
        platform = options['platform']
        event = create_sample_event(project, platform)
        if not event:
            raise CommandError(f'Unable to create an event for platform {platform!r}')
        self.stdout.write(f'Event created: {event.group.get_absolute_url()}')