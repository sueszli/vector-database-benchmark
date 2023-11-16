import graphene
from .....core import EventDeliveryStatus
from .....webhook.event_types import WebhookEventAsyncType
from ....tests.utils import get_graphql_content
EVENT_DELIVERY_QUERY = '\n    query webhook(\n      $id: ID!\n      $first: Int, $last: Int, $after: String, $before: String,\n    ){\n      webhook(id: $id){\n        eventDeliveries(\n            first: $first, last: $last, after: $after, before: $before,\n        ){\n           edges{\n             node{\n               status\n               eventType\n               id\n               payload\n               attempts(first: $first){\n                edges{\n                  node{\n                    id\n                    duration\n                    response\n                    requestHeaders\n                    responseHeaders\n                    responseStatusCode\n                  }\n                }\n              }\n            }\n          }\n        }\n      }\n    }\n'

def test_webhook_delivery_attempt_query(event_attempt, staff_api_client, permission_manage_apps):
    if False:
        for i in range(10):
            print('nop')
    webhook_id = graphene.Node.to_global_id('Webhook', event_attempt.delivery.webhook.pk)
    staff_api_client.user.user_permissions.add(permission_manage_apps)
    variables = {'id': webhook_id, 'first': 3}
    delivery = event_attempt.delivery
    delivery_id = graphene.Node.to_global_id('EventDelivery', delivery.pk)
    response = staff_api_client.post_graphql(EVENT_DELIVERY_QUERY, variables=variables)
    content = get_graphql_content(response)
    delivery_response = content['data']['webhook']['eventDeliveries']['edges'][0]['node']
    attempts_response = delivery_response['attempts']['edges']
    assert delivery_response['id'] == delivery_id
    assert delivery_response['status'] == EventDeliveryStatus.PENDING.upper()
    assert delivery_response['eventType'] == WebhookEventAsyncType.ANY.upper()
    assert delivery_response['payload'] == delivery.payload.payload
    assert len(attempts_response) == 1
    assert attempts_response[0]['node']['response'] == event_attempt.response
    assert attempts_response[0]['node']['duration'] == event_attempt.duration
    assert attempts_response[0]['node']['requestHeaders'] == event_attempt.request_headers
    assert attempts_response[0]['node']['responseHeaders'] == event_attempt.response_headers
    assert attempts_response[0]['node']['responseStatusCode'] == event_attempt.response_status_code