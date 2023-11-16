from ...utils import get_graphql_content
VOUCHER_BULK_DELETE_MUTATION = '\nmutation VoucherBulkDelete ($ids: [ID!]!) {\n  voucherBulkDelete(ids: $ids) {\n    errors {\n      voucherCodes\n      message\n      field\n      code\n    }\n    count\n  }\n}\n'

def voucher_bulk_delete(staff_api_client, voucher_ids):
    if False:
        for i in range(10):
            print('nop')
    variables = {'ids': voucher_ids}
    response = staff_api_client.post_graphql(VOUCHER_BULK_DELETE_MUTATION, variables, check_no_permissions=False)
    content = get_graphql_content(response)
    data = content['data']
    assert data['voucherBulkDelete']['errors'] == []
    return data