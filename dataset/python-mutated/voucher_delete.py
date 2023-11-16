from ...utils import get_graphql_content
VOUCHER_DELETE_MUTATION = '\nmutation VoucherDelete ($id: ID!) {\n  voucherDelete(id: $id) {\n    errors {\n      message\n      code\n      field\n      voucherCodes\n    }\n    voucher {\n      id\n    }\n  }\n}\n'

def voucher_delete(staff_api_client, voucher_id):
    if False:
        for i in range(10):
            print('nop')
    variables = {'id': voucher_id}
    response = staff_api_client.post_graphql(VOUCHER_DELETE_MUTATION, variables, check_no_permissions=False)
    content = get_graphql_content(response)
    data = content['data']['voucherDelete']
    assert data['errors'] == []
    return data