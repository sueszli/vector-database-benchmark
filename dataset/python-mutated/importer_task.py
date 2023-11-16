import json
import requests
import uuid
from django.conf import settings
from django.core.serializers.json import DjangoJSONEncoder
from django.contrib.auth.hashers import make_password
from celery import shared_task
from sentry_sdk import capture_exception
from plane.api.serializers import ImporterSerializer
from plane.db.models import Importer, WorkspaceMember, GithubRepositorySync, GithubRepository, ProjectMember, WorkspaceIntegration, Label, User, IssueProperty
from plane.bgtasks.user_welcome_task import send_welcome_slack

@shared_task
def service_importer(service, importer_id):
    if False:
        print('Hello World!')
    try:
        importer = Importer.objects.get(pk=importer_id)
        importer.status = 'processing'
        importer.save()
        users = importer.data.get('users', [])
        if len(users):
            new_users = User.objects.bulk_create([User(email=user.get('email').strip().lower(), username=uuid.uuid4().hex, password=make_password(uuid.uuid4().hex), is_password_autoset=True) for user in users if user.get('import', False) == 'invite'], batch_size=10, ignore_conflicts=True)
            _ = [send_welcome_slack.delay(str(user.id), True, f'{user.email} was imported to Plane from {service}') for user in new_users]
            workspace_users = User.objects.filter(email__in=[user.get('email').strip().lower() for user in users if user.get('import', False) == 'invite' or user.get('import', False) == 'map'])
            WorkspaceMember.objects.bulk_create([WorkspaceMember(member=user, workspace_id=importer.workspace_id, created_by=importer.created_by) for user in workspace_users], batch_size=100, ignore_conflicts=True)
            ProjectMember.objects.bulk_create([ProjectMember(project_id=importer.project_id, workspace_id=importer.workspace_id, member=user, created_by=importer.created_by) for user in workspace_users], batch_size=100, ignore_conflicts=True)
            IssueProperty.objects.bulk_create([IssueProperty(project_id=importer.project_id, workspace_id=importer.workspace_id, user=user, created_by=importer.created_by) for user in workspace_users], batch_size=100, ignore_conflicts=True)
        if service == 'github' and importer.config.get('sync', False):
            name = importer.metadata.get('name', False)
            url = importer.metadata.get('url', False)
            config = importer.metadata.get('config', {})
            owner = importer.metadata.get('owner', False)
            repository_id = importer.metadata.get('repository_id', False)
            workspace_integration = WorkspaceIntegration.objects.get(workspace_id=importer.workspace_id, integration__provider='github')
            GithubRepositorySync.objects.filter(project_id=importer.project_id).delete()
            GithubRepository.objects.filter(project_id=importer.project_id).delete()
            label = Label.objects.filter(name='GitHub', project_id=importer.project_id).first()
            if label is None:
                label = Label.objects.create(name='GitHub', project_id=importer.project_id, description='Label to sync Plane issues with GitHub issues', color='#003773')
            repo = GithubRepository.objects.create(name=name, url=url, config=config, repository_id=repository_id, owner=owner, project_id=importer.project_id)
            _ = GithubRepositorySync.objects.create(repository=repo, workspace_integration=workspace_integration, actor=workspace_integration.actor, credentials=importer.data.get('credentials', {}), project_id=importer.project_id, label=label)
            _ = ProjectMember.objects.get_or_create(member=workspace_integration.actor, role=20, project_id=importer.project_id)
        if settings.PROXY_BASE_URL:
            headers = {'Content-Type': 'application/json'}
            import_data_json = json.dumps(ImporterSerializer(importer).data, cls=DjangoJSONEncoder)
            _ = requests.post(f'{settings.PROXY_BASE_URL}/hooks/workspaces/{str(importer.workspace_id)}/projects/{str(importer.project_id)}/importers/{str(service)}/', json=import_data_json, headers=headers)
        return
    except Exception as e:
        importer = Importer.objects.get(pk=importer_id)
        importer.status = 'failed'
        importer.save()
        if settings.DEBUG:
            print(e)
        capture_exception(e)
        return