import logging
from rest_framework.request import Request
from rest_framework.response import Response
from sentry import similarity
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases.group import GroupEndpoint
from sentry.api.serializers import serialize
from sentry.models.group import Group
logger = logging.getLogger(__name__)

def _fix_label(label):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(label, tuple):
        return ':'.join(label)
    return label

@region_silo_endpoint
class GroupSimilarIssuesEndpoint(GroupEndpoint):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN}

    def get(self, request: Request, group) -> Response:
        if False:
            return 10
        features = similarity.features
        limit = request.GET.get('limit', None)
        if limit is not None:
            limit = int(limit) + 1
        group_ids = []
        group_scores = []
        for (group_id, scores) in features.compare(group, limit=limit):
            if group_id != group.id:
                group_ids.append(group_id)
                group_scores.append(scores)
        serialized_groups = {int(g['id']): g for g in serialize(list(Group.objects.get_many_from_cache(group_ids)), user=request.user)}
        results = []
        for (group_id, scores) in zip(group_ids, group_scores):
            group = serialized_groups.get(group_id)
            if group is None:
                continue
            results.append((group, {_fix_label(k): v for (k, v) in scores.items()}))
        return Response(results)