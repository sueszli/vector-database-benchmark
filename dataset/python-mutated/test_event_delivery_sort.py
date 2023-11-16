from ....tests.utils import get_graphql_content
EVENT_DELIVERY_SORT_QUERY = '\n    query webhook(\n      $id: ID!\n      $first: Int, $last: Int, $after: String, $before: String,\n      $sortBy: EventDeliverySortingInput\n    ){\n      webhook(id: $id){\n        eventDeliveries(\n            first: $first, last: $last, after: $after, before: $before, sortBy: $sortBy\n        ){\n           edges{\n             node{\n               eventType\n               id\n            }\n          }\n        }\n      }\n    }\n'

def test_webhook_delivery_query_sort_asc(event_deliveries, staff_api_client, permission_manage_apps):
    if False:
        i = 10
        return i + 15
    staff_api_client.user.user_permissions.add(permission_manage_apps)
    variables = {'id': event_deliveries['webhook_id'], 'first': 10, 'sortBy': {'field': 'CREATED_AT', 'direction': 'ASC'}}
    response = staff_api_client.post_graphql(EVENT_DELIVERY_SORT_QUERY, variables=variables)
    content = get_graphql_content(response)
    deliveries_response = content['data']['webhook']['eventDeliveries']['edges']
    assert len(deliveries_response) == 3
    assert deliveries_response[0]['node']['id'] == event_deliveries['delivery_1_id']
    assert deliveries_response[1]['node']['id'] == event_deliveries['delivery_2_id']
    assert deliveries_response[2]['node']['id'] == event_deliveries['delivery_3_id']

def test_webhook_delivery_query_sort_dsc(event_deliveries, staff_api_client, permission_manage_apps):
    if False:
        i = 10
        return i + 15
    staff_api_client.user.user_permissions.add(permission_manage_apps)
    variables = {'id': event_deliveries['webhook_id'], 'first': 10, 'sortBy': {'field': 'CREATED_AT', 'direction': 'DESC'}}
    response = staff_api_client.post_graphql(EVENT_DELIVERY_SORT_QUERY, variables=variables)
    content = get_graphql_content(response)
    deliveries_response = content['data']['webhook']['eventDeliveries']['edges']
    assert len(deliveries_response) == 3
    assert deliveries_response[0]['node']['id'] == event_deliveries['delivery_3_id']
    assert deliveries_response[1]['node']['id'] == event_deliveries['delivery_2_id']
    assert deliveries_response[2]['node']['id'] == event_deliveries['delivery_1_id']
EVENT_DELIVERY_ATTEMPT_SORT_QUERY = '\n    query webhook(\n      $id: ID!\n      $first: Int,\n      $sortBy: EventDeliveryAttemptSortingInput\n    ){\n      webhook(id: $id){\n        eventDeliveries(\n            first: $first\n        ){\n           edges{\n             node{\n               eventType\n               id\n               attempts(first: $first sortBy: $sortBy){\n                 edges{\n                   node{\n                     id\n                     taskId\n                     createdAt\n                     response\n                   }\n                 }\n               }\n             }\n           }\n        }\n      }\n    }\n'

def test_webhook_delivery_attempt_query_sort_asc(delivery_attempts, staff_api_client, permission_manage_apps):
    if False:
        return 10
    staff_api_client.user.user_permissions.add(permission_manage_apps)
    variables = {'id': delivery_attempts['webhook_id'], 'first': 3, 'sortBy': {'field': 'CREATED_AT', 'direction': 'ASC'}}
    response = staff_api_client.post_graphql(EVENT_DELIVERY_ATTEMPT_SORT_QUERY, variables=variables)
    content = get_graphql_content(response)
    deliveries_response = content['data']['webhook']['eventDeliveries']['edges'][0]
    attempts_response = deliveries_response['node']['attempts']['edges']
    assert attempts_response[0]['node']['id'] == delivery_attempts['attempt_1_id']
    assert attempts_response[1]['node']['id'] == delivery_attempts['attempt_2_id']

def test_webhook_delivery_attempt_query_sort_desc(delivery_attempts, staff_api_client, permission_manage_apps):
    if False:
        print('Hello World!')
    staff_api_client.user.user_permissions.add(permission_manage_apps)
    variables = {'id': delivery_attempts['webhook_id'], 'first': 3, 'sortBy': {'field': 'CREATED_AT', 'direction': 'DESC'}}
    response = staff_api_client.post_graphql(EVENT_DELIVERY_ATTEMPT_SORT_QUERY, variables=variables)
    content = get_graphql_content(response)
    deliveries_response = content['data']['webhook']['eventDeliveries']['edges'][0]
    attempts_response = deliveries_response['node']['attempts']['edges']
    assert attempts_response[0]['node']['id'] == delivery_attempts['attempt_3_id']
    assert attempts_response[1]['node']['id'] == delivery_attempts['attempt_2_id']