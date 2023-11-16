from django.db import connections
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases import OrganizationMemberEndpoint
from sentry.api.serializers import serialize
from sentry.models.commit import Commit
from sentry.models.repository import Repository
from sentry.services.hybrid_cloud.user.service import user_service
query = '\nselect c1.*\nfrom sentry_commit c1\njoin (\n    select max(c2.date_added) as date_added, c2.repository_id\n    from sentry_commit as c2\n    join (\n        select distinct commit_id from sentry_releasecommit\n        where organization_id = %%s\n    ) as rc2\n    on c2.id = rc2.commit_id\n    group by c2.repository_id\n) as cmax\non c1.repository_id = cmax.repository_id\nwhere c1.date_added > cmax.date_added\nand c1.author_id IN (\n    select id\n    from sentry_commitauthor\n    where organization_id = %%s\n    and upper(email) IN (%s)\n)\norder by c1.date_added desc\n'
quote_name = connections['default'].ops.quote_name
from rest_framework.request import Request
from rest_framework.response import Response

@region_silo_endpoint
class OrganizationMemberUnreleasedCommitsEndpoint(OrganizationMemberEndpoint):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN}

    def get(self, request: Request, organization, member) -> Response:
        if False:
            for i in range(10):
                print('nop')
        email_list = [e.email for e in filter(lambda x: x.is_verified, user_service.get_user(member.user_id).useremails)]
        if not email_list:
            return self.respond({'commits': [], 'repositories': {}, 'errors': {'missing_emails': True}})
        params = [organization.id, organization.id]
        for e in email_list:
            params.append(e.upper())
        queryset = Commit.objects.raw(query % (', '.join(('%s' for _ in email_list)),), params)
        results = list(queryset)
        if results:
            repos = list(Repository.objects.filter(id__in={r.repository_id for r in results}))
        else:
            repos = []
        return self.respond({'commits': [{'id': c.key, 'message': c.message, 'dateCreated': c.date_added, 'repositoryID': str(c.repository_id)} for c in results], 'repositories': {str(r.id): d for (r, d) in zip(repos, serialize(repos, request.user))}})