from ...utils import get_graphql_content
VOUCHER_CODES_EXPORT_MUTATION = '\nmutation VoucherCreate($input: ExportVoucherCodesInput!) {\n  exportVoucherCodes(input: $input) {\n    errors {\n      code\n      field\n      message\n    }\n    exportFile {\n      status\n      id\n    }\n  }\n}\n'

def raw_export_voucher_codes(staff_api_client, input_data):
    if False:
        return 10
    variables = {'input': input_data}
    response = staff_api_client.post_graphql(VOUCHER_CODES_EXPORT_MUTATION, variables, check_no_permissions=False)
    content = get_graphql_content(response, ignore_errors=True)
    return content

def export_voucher_codes(api_client, input_data):
    if False:
        for i in range(10):
            print('nop')
    response = raw_export_voucher_codes(api_client, input_data)
    assert response['data']['exportVoucherCodes']['errors'] == []
    data = response['data']['exportVoucherCodes']['exportFile']
    assert data['id'] is not None
    return data