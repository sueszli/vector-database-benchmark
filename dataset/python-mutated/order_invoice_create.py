from ...utils import get_graphql_content
ORDER_INVOICE_CREATE_MUTATION = '\nmutation InvoiceCreate($id: ID!, $input: InvoiceCreateInput!){\n  invoiceCreate(orderId: $id,\n  input: $input,\n  ) {\n    errors {\n      code\n      field\n      message\n    }\n    invoice {\n      number\n      url\n      createdAt\n      metadata{\n        key\n        value\n      }\n      privateMetadata{\n        key\n        value\n      }\n    }\n  }\n}\n'

def order_invoice_create(staff_api_client, order_id, input):
    if False:
        print('Hello World!')
    variables = {'id': order_id, 'input': input}
    response = staff_api_client.post_graphql(ORDER_INVOICE_CREATE_MUTATION, variables=variables)
    content = get_graphql_content(response)
    data = content['data']['invoiceCreate']
    assert data['errors'] == []
    return data