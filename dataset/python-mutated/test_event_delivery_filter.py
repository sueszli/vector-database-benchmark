import graphene
from .....core import EventDeliveryStatus
from ....tests.utils import get_graphql_content
EVENT_DELIVERY_FILTER_QUERY = '\n    query webhook(\n      $id: ID!\n      $first: Int, $last: Int, $after: String, $before: String,\n      $filters: EventDeliveryFilterInput!\n    ){\n      webhook(id: $id){\n        eventDeliveries(\n            first: $first, last: $last, after: $after, before: $before,\n            filter: $filters\n        ){\n           edges{\n             node{\n               status\n               eventType\n               id\n            }\n          }\n        }\n      }\n    }\n'

def test_delivery_status_filter(event_delivery, staff_api_client, permission_manage_apps):
    if False:
        i = 10
        return i + 15
    webhook_id = graphene.Node.to_global_id('Webhook', event_delivery.webhook.pk)
    staff_api_client.user.user_permissions.add(permission_manage_apps)
    variables = {'filters': {'status': EventDeliveryStatus.PENDING.upper()}, 'id': webhook_id, 'first': 3}
    response = staff_api_client.post_graphql(EVENT_DELIVERY_FILTER_QUERY, variables=variables)
    content = get_graphql_content(response)
    delivery_response = content['data']['webhook']['eventDeliveries']
    assert delivery_response['edges'][0]['node']['id'] == graphene.Node.to_global_id('EventDelivery', event_delivery.pk)

def test_delivery_status_filter_no_results(event_delivery, staff_api_client, permission_manage_apps):
    if False:
        return 10
    webhook_id = graphene.Node.to_global_id('Webhook', event_delivery.webhook.pk)
    staff_api_client.user.user_permissions.add(permission_manage_apps)
    variables = {'filters': {'status': EventDeliveryStatus.SUCCESS.upper()}, 'id': webhook_id, 'first': 3}
    response = staff_api_client.post_graphql(EVENT_DELIVERY_FILTER_QUERY, variables=variables)
    content = get_graphql_content(response)
    assert len(content['data']['webhook']['eventDeliveries']['edges']) == 0