from ...utils import get_graphql_content
CHECKOUT_LINES_UPDATE_MUTATION = '\nmutation checkoutLinesUpdate($checkoutId: ID!, $lines: [CheckoutLineUpdateInput!]! ){\n  checkoutLinesUpdate(lines: $lines, checkoutId: $checkoutId) {\n    checkout {\n      lines {\n        quantity\n        variant {\n          id\n          quantityLimitPerCustomer\n        }\n      }\n    }\n    errors {\n      field\n      lines\n      message\n      variants\n      code\n    }\n  }\n}\n'

def checkout_lines_update(staff_api_client, checkout_id, lines):
    if False:
        i = 10
        return i + 15
    variables = {'checkoutId': checkout_id, 'lines': lines}
    response = staff_api_client.post_graphql(CHECKOUT_LINES_UPDATE_MUTATION, variables)
    content = get_graphql_content(response)
    return content['data']['checkoutLinesUpdate']