from ...utils import get_graphql_content
CUSTOMER_BULK_UPDATE_MUTATION = '\nmutation CustomerBulkUpdate($errorPolicy: ErrorPolicyEnum, $customers: [CustomerBulkUpdateInput!]!) {\n  customerBulkUpdate(errorPolicy: $errorPolicy, customers: $customers) {\n    count\n    results {\n      customer {\n        id\n        email\n        metadata {\n          key\n          value\n        }\n        privateMetadata {\n          key\n          value\n        }\n      }\n      errors {\n        path\n        message\n        code\n      }\n    }\n    errors {\n      path\n      message\n      code\n    }\n  }\n}\n'

def customer_bulk_update(staff_api_client, customers, error_policy='REJECT_EVERYTHING'):
    if False:
        return 10
    variables = {'customers': customers, 'errorPolicy': error_policy}
    response = staff_api_client.post_graphql(CUSTOMER_BULK_UPDATE_MUTATION, variables)
    content = get_graphql_content(response)
    data = content['data']['customerBulkUpdate']
    return data