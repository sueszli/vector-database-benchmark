from ...utils import get_graphql_content
VOUCHER_UPDATE_MUTATION = '\nmutation VoucherUpdate($id:ID! $input: VoucherInput!) {\n  voucherUpdate(id: $id, input: $input) {\n    errors {\n      message\n      code\n      field\n    }\n    voucher {\n      id\n      applyOncePerCustomer\n      applyOncePerOrder\n      endDate\n      onlyForStaff\n      startDate\n      type\n      usageLimit\n      used\n      minCheckoutItemsQuantity\n      minSpent {\n        amount\n      }\n      code\n      discountValueType\n      discountValue\n    }\n  }\n}\n'

def raw_update_voucher(staff_api_client, voucher_id, input_data):
    if False:
        print('Hello World!')
    variables = {'id': voucher_id, 'input': input_data}
    response = staff_api_client.post_graphql(VOUCHER_UPDATE_MUTATION, variables, check_no_permissions=False)
    content = get_graphql_content(response)
    data = content['data']['voucherUpdate']
    return data

def update_voucher(staff_api_client, voucher_id, input_data):
    if False:
        i = 10
        return i + 15
    response = raw_update_voucher(staff_api_client, voucher_id, input_data)
    assert response['errors'] == []
    data = response['voucher']
    assert data['id'] is not None
    return data