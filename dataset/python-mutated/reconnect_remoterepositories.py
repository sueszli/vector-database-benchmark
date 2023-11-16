import json
from django.db.models import Q, Subquery
from django.core.management.base import BaseCommand
from readthedocs.oauth.models import RemoteRepository
from readthedocs.oauth.services import registry
from readthedocs.oauth.services.base import SyncServiceError
from readthedocs.projects.models import Project
from readthedocs.organizations.models import Organization

class Command(BaseCommand):
    help = 'Re-connect RemoteRepository to Project'

    def add_arguments(self, parser):
        if False:
            return 10
        parser.add_argument('organization', nargs='+', type=str)
        parser.add_argument('--no-dry-run', action='store_true', default=False, help='Update database with the changes proposed.')
        parser.add_argument('--only-owners', action='store_true', default=False, help='Connect repositories only to organization owners.')
        parser.add_argument('--force-owners-social-resync', action='store_true', default=False, help='Force to re-sync RemoteRepository for organization owners.')

    def _force_owners_social_resync(self, organization):
        if False:
            i = 10
            return i + 15
        for owner in organization.owners.all():
            for service_cls in registry:
                for service in service_cls.for_user(owner):
                    try:
                        service.sync()
                    except SyncServiceError:
                        print(f'Service {service} failed while syncing. Skipping...')

    def _connect_repositories(self, organization, no_dry_run, only_owners):
        if False:
            return 10
        connected_projects = []
        remote_query = Q(ssh_url__in=Subquery(organization.projects.values('repo'))) | Q(clone_url__in=Subquery(organization.projects.values('repo')))
        for remote in RemoteRepository.objects.filter(remote_query).order_by('created'):
            admin = json.loads(remote.json).get('permissions', {}).get('admin')
            if only_owners and remote.users.first() not in organization.owners.all():
                continue
            if not admin:
                continue
            if not organization.users.filter(username=remote.users.first().username).exists():
                continue
            projects = Project.objects.filter(Q(repo=remote.ssh_url) | Q(repo=remote.clone_url), organizations__in=[organization.pk], remote_repository__isnull=True).exclude(slug__in=connected_projects)
            for project in projects:
                connected_projects.append(project.slug)
                if no_dry_run:
                    remote.project = project
                    remote.save()
                print(f'{project.slug: <40} {remote.pk: <10} {remote.html_url: <60} {remote.users.first().username: <20} {admin: <5}')
        print('Total:', len(connected_projects))
        if not no_dry_run:
            print('Changes WERE NOT applied to the database. Run it with --no-dry-run to save the changes.')

    def handle(self, *args, **options):
        if False:
            print('Hello World!')
        no_dry_run = options.get('no_dry_run')
        only_owners = options.get('only_owners')
        force_owners_social_resync = options.get('force_owners_social_resync')
        for organization in options.get('organization'):
            try:
                organization = Organization.objects.get(slug=organization)
                if force_owners_social_resync:
                    self._force_owners_social_resync(organization)
                self._connect_repositories(organization, no_dry_run, only_owners)
            except Organization.DoesNotExist:
                print(f'Organization does not exist. organization={organization}')