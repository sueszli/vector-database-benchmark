from django.db import IntegrityError
from django.http import Http404
from sentry.models.commitauthor import CommitAuthor
from sentry.models.pullrequest import PullRequest
from sentry.models.repository import Repository
from sentry.services.hybrid_cloud.user.service import user_service
from . import Webhook, get_external_id

class PullRequestEventWebhook(Webhook):

    def __call__(self, event, organization):
        if False:
            print('Hello World!')
        is_apps = 'installation' in event
        try:
            repo = Repository.objects.get(organization_id=organization.id, provider='github_apps' if is_apps else 'github', external_id=str(event['repository']['id']))
        except Repository.DoesNotExist:
            raise Http404()
        if repo.config.get('name') != event['repository']['full_name']:
            repo.config['name'] = event['repository']['full_name']
            repo.save()
        pull_request = event['pull_request']
        number = pull_request['number']
        title = pull_request['title']
        body = pull_request['body']
        user = pull_request['user']
        merge_commit_sha = pull_request['merge_commit_sha'] if pull_request['merged'] else None
        author_email = '{}@localhost'.format(user['login'][:65])
        try:
            commit_author = CommitAuthor.objects.get(external_id=get_external_id(user['login']), organization_id=organization.id)
            author_email = commit_author.email
        except CommitAuthor.DoesNotExist:
            rpc_user = user_service.get_user_by_social_auth(organization_id=organization.id, provider='github', uid=user['id'])
            if rpc_user is not None:
                author_email = rpc_user.email
        try:
            author = CommitAuthor.objects.get(organization_id=organization.id, external_id=get_external_id(user['login']))
        except CommitAuthor.DoesNotExist:
            try:
                author = CommitAuthor.objects.get(organization_id=organization.id, email=author_email)
            except CommitAuthor.DoesNotExist:
                author = CommitAuthor.objects.create(organization_id=organization.id, email=author_email, external_id=get_external_id(user['login']), name=user['login'][:128])
        author.preload_users()
        try:
            PullRequest.objects.update_or_create(organization_id=organization.id, repository_id=repo.id, key=number, defaults={'organization_id': organization.id, 'title': title, 'author': author, 'message': body, 'merge_commit_sha': merge_commit_sha})
        except IntegrityError:
            pass