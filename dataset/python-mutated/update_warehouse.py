from ...utils import get_graphql_content
WAREHOUSE_UPDATE_MUTATION = '\nmutation updateWarehouse($id: ID! $input: WarehouseUpdateInput!) {\n  updateWarehouse(id: $id, input: $input) {\n    errors {\n      message\n      field\n      code\n    }\n    warehouse {\n      id\n      name\n      slug\n      clickAndCollectOption\n      isPrivate\n    }\n  }\n}\n'

def update_warehouse(staff_api_client, warehouse_id, is_private=False, click_and_collect_option='DISABLED'):
    if False:
        return 10
    variables = {'id': warehouse_id, 'input': {'isPrivate': is_private, 'clickAndCollectOption': click_and_collect_option}}
    response = staff_api_client.post_graphql(WAREHOUSE_UPDATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['updateWarehouse']['errors'] == []
    data = content['data']['updateWarehouse']['warehouse']
    assert data['isPrivate'] == is_private
    assert data['clickAndCollectOption'] == click_and_collect_option
    return data