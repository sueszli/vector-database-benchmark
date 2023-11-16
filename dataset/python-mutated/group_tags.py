from __future__ import annotations
from typing import TYPE_CHECKING, Any
from rest_framework.request import Request
from rest_framework.response import Response
from sentry import tagstore
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases.group import GroupEndpoint
from sentry.api.helpers.environments import get_environments
from sentry.api.helpers.mobile import get_readable_device_name
from sentry.api.serializers import serialize
from sentry.search.utils import DEVICE_CLASS
if TYPE_CHECKING:
    from sentry.models.group import Group

@region_silo_endpoint
class GroupTagsEndpoint(GroupEndpoint):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN}

    def get(self, request: Request, group: Group) -> Response:
        if False:
            for i in range(10):
                print('nop')
        keys = [tagstore.prefix_reserved_key(k) for k in request.GET.getlist('key') if k] or None
        limit = request.GET.get('limit')
        if limit is not None:
            value_limit = int(limit)
        elif keys:
            value_limit = 9
        else:
            value_limit = 10
        environment_ids = [e.id for e in get_environments(request, group.project.organization)]
        tag_keys = tagstore.get_group_tag_keys_and_top_values(group, environment_ids, keys=keys, value_limit=value_limit, tenant_ids={'organization_id': group.project.organization_id})
        data = serialize(tag_keys, request.user)
        show_readable_tag_values = request.GET.get('readable')
        if show_readable_tag_values:
            add_readable_tag_values(data)
        map_device_class(data)
        return Response(data)

def add_readable_tag_values(data: Any) -> None:
    if False:
        while True:
            i = 10
    for device_tag in data:
        if device_tag['key'] == 'device':
            for top_device in device_tag['topValues']:
                readable_value = get_readable_device_name(top_device['value'])
                if readable_value:
                    top_device['readable'] = readable_value
            break

def map_device_class(data: Any) -> None:
    if False:
        for i in range(10):
            print('nop')
    for tag in data:
        if tag['key'] == 'device.class':
            for top_device_class in tag['topValues']:
                for (key, value) in DEVICE_CLASS.items():
                    if top_device_class['value'] in value:
                        top_device_class['value'] = key
                        top_device_class['name'] = key
                        break
            break