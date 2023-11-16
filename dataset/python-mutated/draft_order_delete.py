from saleor.graphql.tests.utils import get_graphql_content
DRAFT_ORDER_DELETE_MUTATION = '\nmutation DraftOrderDelete($id: ID!) {\n  draftOrderDelete(id: $id) {\n    errors {\n\t\tmessage\n        field\n    }\n    order {\n\t\tstatus\n    }\n  }\n}\n'

def draft_order_delete(api_client, id):
    if False:
        while True:
            i = 10
    variables = {'id': id}
    response = api_client.post_graphql(DRAFT_ORDER_DELETE_MUTATION, variables=variables)
    content = get_graphql_content(response)
    data = content['data']['draftOrderDelete']
    errors = data['errors']
    assert errors == []
    return data