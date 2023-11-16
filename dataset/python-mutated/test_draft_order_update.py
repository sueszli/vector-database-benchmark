import graphene
from prices import TaxedMoney
from .....core.prices import quantize_price
from .....core.taxes import zero_money
from .....discount import DiscountType, DiscountValueType
from .....order import OrderStatus
from .....order.error_codes import OrderErrorCode
from .....order.models import OrderEvent
from ....tests.utils import assert_no_permission, get_graphql_content
DRAFT_ORDER_UPDATE_MUTATION = '\n        mutation draftUpdate(\n        $id: ID!,\n        $input: DraftOrderInput!,\n        ) {\n            draftOrderUpdate(\n                id: $id,\n                input: $input\n            ) {\n                errors {\n                    field\n                    code\n                    message\n                }\n                order {\n                    userEmail\n                    externalReference\n                    channel {\n                        id\n                    }\n                    total {\n                        net {\n                            amount\n                        }\n                    }\n                    undiscountedTotal {\n                        net {\n                            amount\n                        }\n                    }\n                    billingAddress{\n                        city\n                        streetAddress1\n                        postalCode\n                        metadata {\n                            key\n                            value\n                        }\n                    }\n                    voucher {\n                        code\n                    }\n                    voucherCode\n                    shippingAddress{\n                        city\n                        streetAddress1\n                        postalCode\n                        metadata {\n                            key\n                            value\n                        }\n                    }\n                    total {\n                        gross {\n                            amount\n                        }\n                    }\n                    undiscountedTotal {\n                        gross {\n                            amount\n                        }\n                    }\n                }\n            }\n        }\n        '

def test_draft_order_update_existing_channel_id(staff_api_client, permission_group_manage_orders, order_with_lines, channel_PLN):
    if False:
        i = 10
        return i + 15
    order = order_with_lines
    order.status = OrderStatus.DRAFT
    order.save()
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    query = DRAFT_ORDER_UPDATE_MUTATION
    order_id = graphene.Node.to_global_id('Order', order.id)
    channel_id = graphene.Node.to_global_id('Channel', channel_PLN.id)
    variables = {'id': order_id, 'input': {'channelId': channel_id}}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    error = content['data']['draftOrderUpdate']['errors'][0]
    assert error['code'] == OrderErrorCode.NOT_EDITABLE.name
    assert error['field'] == 'channelId'

def test_draft_order_update_voucher_not_available(staff_api_client, permission_group_manage_orders, order_with_lines, voucher):
    if False:
        while True:
            i = 10
    order = order_with_lines
    order.status = OrderStatus.DRAFT
    order.save()
    assert order.voucher is None
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    query = DRAFT_ORDER_UPDATE_MUTATION
    order_id = graphene.Node.to_global_id('Order', order.id)
    voucher_id = graphene.Node.to_global_id('Voucher', voucher.id)
    voucher.channel_listings.all().delete()
    variables = {'id': order_id, 'input': {'voucher': voucher_id}}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    error = content['data']['draftOrderUpdate']['errors'][0]
    assert error['code'] == OrderErrorCode.NOT_AVAILABLE_IN_CHANNEL.name
    assert error['field'] == 'voucher'

def test_draft_order_update_with_voucher(staff_api_client, permission_group_manage_orders, draft_order, voucher, graphql_address_data):
    if False:
        while True:
            i = 10
    order = draft_order
    assert not order.voucher
    assert not order.voucher_code
    assert not order.customer_note
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    query = DRAFT_ORDER_UPDATE_MUTATION
    order_id = graphene.Node.to_global_id('Order', order.id)
    voucher_id = graphene.Node.to_global_id('Voucher', voucher.id)
    voucher_listing = voucher.channel_listings.get(channel=order.channel)
    customer_note = 'Test customer note'
    external_reference = 'test-ext-ref'
    order_total = order.total_net_amount
    variables = {'id': order_id, 'input': {'voucher': voucher_id, 'customerNote': customer_note, 'externalReference': external_reference, 'shippingAddress': graphql_address_data, 'billingAddress': graphql_address_data}}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['draftOrderUpdate']
    assert data['order']['voucher']['code'] == voucher.code
    assert data['order']['voucherCode'] == voucher.code
    stored_metadata = {'public': 'public_value'}
    assert data['order']['billingAddress']['metadata'] == graphql_address_data['metadata']
    assert data['order']['shippingAddress']['metadata'] == graphql_address_data['metadata']
    assert data['order']['undiscountedTotal']['net']['amount'] == order_total
    assert data['order']['total']['net']['amount'] == order_total - voucher_listing.discount_value
    assert not data['errors']
    order.refresh_from_db()
    assert order.billing_address.metadata == stored_metadata
    assert order.shipping_address.metadata == stored_metadata
    assert order.voucher_code == voucher.code
    assert order.customer_note == customer_note
    assert order.search_vector
    assert data['order']['externalReference'] == external_reference == order.external_reference
    assert order.discounts.count() == 1
    order_discount = order.discounts.first()
    assert order_discount.voucher == voucher
    assert order_discount.type == DiscountType.VOUCHER
    assert order_discount.value_type == DiscountValueType.FIXED
    assert order_discount.value == voucher_listing.discount_value
    assert order_discount.amount_value == voucher_listing.discount_value

def test_draft_order_update_clear_voucher(staff_api_client, permission_group_manage_orders, draft_order, voucher):
    if False:
        print('Hello World!')
    order = draft_order
    order.voucher = voucher
    order.save(update_fields=['voucher'])
    voucher_listing = voucher.channel_listings.get(channel=order.channel)
    discount_amount = voucher_listing.discount_value
    order.discounts.create(voucher=voucher, value=discount_amount, type=DiscountType.VOUCHER)
    order.total_gross_amount -= discount_amount
    order.total_net_amount -= discount_amount
    order.save(update_fields=['total_net_amount', 'total_gross_amount'])
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    query = DRAFT_ORDER_UPDATE_MUTATION
    order_id = graphene.Node.to_global_id('Order', order.id)
    order_total = order.undiscounted_total_net_amount
    variables = {'id': order_id, 'input': {'voucher': None}}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['draftOrderUpdate']
    assert data['order']['undiscountedTotal']['net']['amount'] == order_total
    assert data['order']['total']['net']['amount'] == order_total
    assert not data['errors']
    order.refresh_from_db()
    assert not order.voucher
    assert order.search_vector
    assert not order.discounts.count()

def test_draft_order_update_clear_voucher_code(staff_api_client, permission_group_manage_orders, draft_order, voucher):
    if False:
        for i in range(10):
            print('nop')
    order = draft_order
    order.voucher = voucher
    order.save(update_fields=['voucher'])
    voucher_listing = voucher.channel_listings.get(channel=order.channel)
    discount_amount = voucher_listing.discount_value
    order.discounts.create(voucher=voucher, value=discount_amount, type=DiscountType.VOUCHER)
    order.total_gross_amount -= discount_amount
    order.total_net_amount -= discount_amount
    order.save(update_fields=['total_net_amount', 'total_gross_amount'])
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    query = DRAFT_ORDER_UPDATE_MUTATION
    order_id = graphene.Node.to_global_id('Order', order.id)
    order_total = order.undiscounted_total_net_amount
    variables = {'id': order_id, 'input': {'voucherCode': None}}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['draftOrderUpdate']
    assert data['order']['undiscountedTotal']['net']['amount'] == order_total
    assert data['order']['total']['net']['amount'] == order_total
    assert not data['errors']
    order.refresh_from_db()
    assert not order.voucher
    assert order.search_vector
    assert not order.discounts.count()

def test_draft_order_update_with_voucher_and_voucher_code(staff_api_client, permission_group_manage_orders, draft_order, voucher, graphql_address_data):
    if False:
        for i in range(10):
            print('nop')
    order = draft_order
    assert not order.voucher
    assert not order.voucher_code
    assert not order.customer_note
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    query = DRAFT_ORDER_UPDATE_MUTATION
    order_id = graphene.Node.to_global_id('Order', order.id)
    voucher_id = graphene.Node.to_global_id('Voucher', voucher.id)
    customer_note = 'Test customer note'
    external_reference = 'test-ext-ref'
    variables = {'id': order_id, 'input': {'voucher': voucher_id, 'voucherCode': voucher.codes.first().code, 'customerNote': customer_note, 'externalReference': external_reference, 'shippingAddress': graphql_address_data, 'billingAddress': graphql_address_data}}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    error = content['data']['draftOrderUpdate']['errors'][0]
    assert error['field'] == 'voucher'
    assert error['code'] == OrderErrorCode.INVALID.name
    assert error['message'] == 'You cannot use both a voucher and a voucher code for the same order. Please choose one.'

def test_draft_order_update_with_voucher_including_drafts_in_voucher_usage(staff_api_client, permission_group_manage_orders, draft_order, voucher):
    if False:
        i = 10
        return i + 15
    order = draft_order
    assert not order.voucher
    assert not order.voucher_code
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    query = DRAFT_ORDER_UPDATE_MUTATION
    order_id = graphene.Node.to_global_id('Order', order.id)
    voucher_id = graphene.Node.to_global_id('Voucher', voucher.id)
    voucher_listing = voucher.channel_listings.get(channel=order.channel)
    order_total = order.total_net_amount
    channel = order.channel
    channel.include_draft_order_in_voucher_usage = True
    channel.save(update_fields=['include_draft_order_in_voucher_usage'])
    voucher.single_use = True
    voucher.save(update_fields=['single_use'])
    variables = {'id': order_id, 'input': {'voucher': voucher_id}}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['draftOrderUpdate']
    assert data['order']['voucher']['code'] == voucher.code
    assert data['order']['voucherCode'] == voucher.code
    assert data['order']['undiscountedTotal']['net']['amount'] == order_total
    assert data['order']['total']['net']['amount'] == order_total - voucher_listing.discount_value
    assert not data['errors']
    order.refresh_from_db()
    assert order.voucher_code == voucher.code
    assert order.search_vector
    assert order.discounts.count() == 1
    order_discount = order.discounts.first()
    assert order_discount.voucher == voucher
    assert order_discount.type == DiscountType.VOUCHER
    assert order_discount.value_type == DiscountValueType.FIXED
    assert order_discount.value == voucher_listing.discount_value
    assert order_discount.amount_value == voucher_listing.discount_value
    code_instance = voucher.codes.first()
    assert code_instance.is_active is False

def test_draft_order_update_with_voucher_code_including_drafts_in_voucher_usage(staff_api_client, permission_group_manage_orders, draft_order, voucher):
    if False:
        return 10
    order = draft_order
    assert not order.voucher
    assert not order.voucher_code
    assert not order.customer_note
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    query = DRAFT_ORDER_UPDATE_MUTATION
    order_id = graphene.Node.to_global_id('Order', order.id)
    voucher_listing = voucher.channel_listings.get(channel=order.channel)
    order_total = order.total_net_amount
    channel = order.channel
    channel.include_draft_order_in_voucher_usage = True
    channel.save(update_fields=['include_draft_order_in_voucher_usage'])
    voucher.single_use = True
    voucher.save(update_fields=['single_use'])
    code_instance = voucher.codes.first()
    variables = {'id': order_id, 'input': {'voucherCode': code_instance.code}}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['draftOrderUpdate']
    assert data['order']['voucher']['code'] == voucher.code
    assert data['order']['voucherCode'] == voucher.code
    assert data['order']['undiscountedTotal']['net']['amount'] == order_total
    assert data['order']['total']['net']['amount'] == order_total - voucher_listing.discount_value
    assert not data['errors']
    order.refresh_from_db()
    assert order.voucher_code == voucher.code
    assert order.search_vector
    assert order.discounts.count() == 1
    order_discount = order.discounts.first()
    assert order_discount.voucher == voucher
    assert order_discount.type == DiscountType.VOUCHER
    assert order_discount.value_type == DiscountValueType.FIXED
    assert order_discount.value == voucher_listing.discount_value
    assert order_discount.amount_value == voucher_listing.discount_value
    code_instance.refresh_from_db()
    assert code_instance.is_active is False

def test_draft_order_update_voucher_including_drafts_in_voucher_usage_invalid_code(staff_api_client, permission_group_manage_orders, order_with_lines, voucher):
    if False:
        i = 10
        return i + 15
    order = order_with_lines
    order.status = OrderStatus.DRAFT
    order.save()
    assert order.voucher is None
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    channel = order.channel
    channel.include_draft_order_in_voucher_usage = True
    channel.save(update_fields=['include_draft_order_in_voucher_usage'])
    voucher.single_use = True
    voucher.save(update_fields=['single_use'])
    code = voucher.codes.first()
    code.is_active = False
    code.save(update_fields=['is_active'])
    query = DRAFT_ORDER_UPDATE_MUTATION
    order_id = graphene.Node.to_global_id('Order', order.id)
    voucher_id = graphene.Node.to_global_id('Voucher', voucher.id)
    variables = {'id': order_id, 'input': {'voucher': voucher_id}}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    error = content['data']['draftOrderUpdate']['errors'][0]
    assert error['code'] == OrderErrorCode.INVALID_VOUCHER.name
    assert error['field'] == 'voucher'

def test_draft_order_update_with_non_draft_order(staff_api_client, permission_group_manage_orders, order_with_lines, voucher):
    if False:
        return 10
    order = order_with_lines
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    query = DRAFT_ORDER_UPDATE_MUTATION
    order_id = graphene.Node.to_global_id('Order', order.id)
    voucher_id = graphene.Node.to_global_id('Voucher', voucher.id)
    customer_note = 'Test customer note'
    variables = {'id': order_id, 'input': {'voucher': voucher_id, 'customerNote': customer_note}}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    error = content['data']['draftOrderUpdate']['errors'][0]
    assert error['field'] == 'id'
    assert error['code'] == OrderErrorCode.INVALID.name

def test_draft_order_update_invalid_address(staff_api_client, permission_group_manage_orders, draft_order, voucher, graphql_address_data):
    if False:
        while True:
            i = 10
    order = draft_order
    assert not order.voucher
    assert not order.customer_note
    graphql_address_data['postalCode'] = 'TEST TEST invalid postal code 12345'
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    query = DRAFT_ORDER_UPDATE_MUTATION
    order_id = graphene.Node.to_global_id('Order', order.id)
    voucher_id = graphene.Node.to_global_id('Voucher', voucher.id)
    variables = {'id': order_id, 'input': {'voucher': voucher_id, 'shippingAddress': graphql_address_data}}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['draftOrderUpdate']
    assert len(data['errors']) == 2
    assert not data['order']
    assert {error['code'] for error in data['errors']} == {OrderErrorCode.INVALID.name, OrderErrorCode.REQUIRED.name}
    assert {error['field'] for error in data['errors']} == {'postalCode'}

def test_draft_order_update_by_user_no_channel_access(staff_api_client, permission_group_all_perms_channel_USD_only, draft_order, channel_PLN):
    if False:
        while True:
            i = 10
    permission_group_all_perms_channel_USD_only.user_set.add(staff_api_client.user)
    order = draft_order
    order.channel = channel_PLN
    order.save(update_fields=['channel'])
    assert not order.customer_note
    query = DRAFT_ORDER_UPDATE_MUTATION
    order_id = graphene.Node.to_global_id('Order', order.id)
    customer_note = 'Test customer note'
    variables = {'id': order_id, 'input': {'customerNote': customer_note}}
    response = staff_api_client.post_graphql(query, variables)
    assert_no_permission(response)

def test_draft_order_update_by_app(app_api_client, permission_manage_orders, draft_order, channel_PLN):
    if False:
        while True:
            i = 10
    order = draft_order
    order.channel = channel_PLN
    order.save(update_fields=['channel'])
    assert not order.customer_note
    query = DRAFT_ORDER_UPDATE_MUTATION
    order_id = graphene.Node.to_global_id('Order', order.id)
    customer_note = 'Test customer note'
    variables = {'id': order_id, 'input': {'customerNote': customer_note}}
    response = app_api_client.post_graphql(query, variables, permissions=(permission_manage_orders,))
    content = get_graphql_content(response)
    data = content['data']['draftOrderUpdate']
    assert not data['errors']
    order.refresh_from_db()
    assert order.customer_note == customer_note
    assert order.search_vector

def test_draft_order_update_doing_nothing_generates_no_events(staff_api_client, permission_group_manage_orders, order_with_lines):
    if False:
        i = 10
        return i + 15
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    assert not OrderEvent.objects.exists()
    query = '\n        mutation draftUpdate($id: ID!) {\n            draftOrderUpdate(id: $id, input: {}) {\n                errors {\n                    field\n                    message\n                }\n            }\n        }\n        '
    order_id = graphene.Node.to_global_id('Order', order_with_lines.id)
    response = staff_api_client.post_graphql(query, {'id': order_id})
    get_graphql_content(response)
    assert not OrderEvent.objects.exists()

def test_draft_order_update_free_shipping_voucher(staff_api_client, permission_group_manage_orders, draft_order, voucher_free_shipping):
    if False:
        while True:
            i = 10
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    order = draft_order
    assert not order.voucher
    query = '\n        mutation draftUpdate(\n            $id: ID!\n            $voucher: ID!\n        ) {\n            draftOrderUpdate(\n                id: $id\n                input: {\n                    voucher: $voucher\n                }\n            ) {\n                errors {\n                    field\n                    message\n                    code\n                }\n                order {\n                    id\n                }\n            }\n        }\n        '
    voucher = voucher_free_shipping
    order_id = graphene.Node.to_global_id('Order', order.id)
    voucher_id = graphene.Node.to_global_id('Voucher', voucher.id)
    variables = {'id': order_id, 'voucher': voucher_id}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['draftOrderUpdate']
    assert not data['errors']
    assert data['order']['id'] == variables['id']
    order.refresh_from_db()
    assert order.voucher
DRAFT_ORDER_UPDATE_USER_EMAIL_MUTATION = '\n    mutation draftUpdate(\n        $id: ID!\n        $userEmail: String!\n    ) {\n        draftOrderUpdate(\n            id: $id\n            input: {\n                userEmail: $userEmail\n            }\n        ) {\n            errors {\n                field\n                message\n                code\n            }\n            order {\n                id\n            }\n        }\n    }\n    '

def test_draft_order_update_when_not_existing_customer_email_provided(staff_api_client, permission_group_manage_orders, draft_order):
    if False:
        return 10
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    order = draft_order
    assert order.user
    query = DRAFT_ORDER_UPDATE_USER_EMAIL_MUTATION
    order_id = graphene.Node.to_global_id('Order', order.id)
    email = 'notexisting@example.com'
    variables = {'id': order_id, 'userEmail': email}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['draftOrderUpdate']
    order.refresh_from_db()
    assert not data['errors']
    assert not order.user
    assert order.user_email == email

def test_draft_order_update_assign_user_when_existing_customer_email_provided(staff_api_client, permission_group_manage_orders, draft_order):
    if False:
        i = 10
        return i + 15
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    order = draft_order
    user = order.user
    user_email = user.email
    order.user = None
    order.save(update_fields=['user'])
    assert not order.user
    query = DRAFT_ORDER_UPDATE_USER_EMAIL_MUTATION
    order_id = graphene.Node.to_global_id('Order', order.id)
    variables = {'id': order_id, 'userEmail': user_email}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['draftOrderUpdate']
    order.refresh_from_db()
    assert not data['errors']
    assert order.user == user
    assert order.user_email == user_email
DRAFT_ORDER_UPDATE_BY_EXTERNAL_REFERENCE = '\n    mutation draftUpdate(\n        $id: ID\n        $externalReference: String\n        $input: DraftOrderInput!\n    ) {\n        draftOrderUpdate(\n            id: $id\n            externalReference: $externalReference\n            input: $input\n        ) {\n            errors {\n                field\n                message\n                code\n            }\n            order {\n                id\n                externalReference\n                voucher {\n                    id\n                }\n            }\n        }\n    }\n    '

def test_draft_order_update_by_external_reference(staff_api_client, permission_group_manage_orders, draft_order, voucher_free_shipping):
    if False:
        return 10
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    query = DRAFT_ORDER_UPDATE_BY_EXTERNAL_REFERENCE
    order = draft_order
    assert not order.voucher
    voucher = voucher_free_shipping
    voucher_id = graphene.Node.to_global_id('Voucher', voucher.id)
    ext_ref = 'test-ext-ref'
    order.external_reference = ext_ref
    order.save(update_fields=['external_reference'])
    variables = {'externalReference': ext_ref, 'input': {'voucher': voucher_id}}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['draftOrderUpdate']
    assert not data['errors']
    assert data['order']['externalReference'] == ext_ref
    assert data['order']['id'] == graphene.Node.to_global_id('Order', order.id)
    assert data['order']['voucher']['id'] == voucher_id
    order.refresh_from_db()
    assert order.voucher

def test_draft_order_update_by_both_id_and_external_reference(staff_api_client, permission_group_manage_orders, voucher_free_shipping):
    if False:
        return 10
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    query = DRAFT_ORDER_UPDATE_BY_EXTERNAL_REFERENCE
    variables = {'id': 'test-id', 'externalReference': 'test-ext-ref', 'input': {}}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['draftOrderUpdate']
    assert not data['order']
    assert data['errors'][0]['message'] == "Argument 'id' cannot be combined with 'external_reference'"

def test_draft_order_update_by_external_reference_not_existing(staff_api_client, permission_group_manage_orders, voucher_free_shipping):
    if False:
        i = 10
        return i + 15
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    query = DRAFT_ORDER_UPDATE_BY_EXTERNAL_REFERENCE
    ext_ref = 'non-existing-ext-ref'
    variables = {'externalReference': ext_ref, 'input': {}}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['draftOrderUpdate']
    assert not data['order']
    assert data['errors'][0]['message'] == f"Couldn't resolve to a node: {ext_ref}"

def test_draft_order_update_with_non_unique_external_reference(staff_api_client, permission_group_manage_orders, draft_order, order_list):
    if False:
        return 10
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    query = DRAFT_ORDER_UPDATE_BY_EXTERNAL_REFERENCE
    draft_order_id = graphene.Node.to_global_id('Order', draft_order.pk)
    ext_ref = 'test-ext-ref'
    order = order_list[1]
    order.external_reference = ext_ref
    order.save(update_fields=['external_reference'])
    variables = {'id': draft_order_id, 'input': {'externalReference': ext_ref}}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    error = content['data']['draftOrderUpdate']['errors'][0]
    assert error['field'] == 'externalReference'
    assert error['code'] == OrderErrorCode.UNIQUE.name
    assert error['message'] == 'Order with this External reference already exists.'
DRAFT_ORDER_UPDATE_SHIPPING_METHOD_MUTATION = '\n    mutation draftUpdate($id: ID!, $shippingMethod: ID){\n        draftOrderUpdate(\n            id: $id,\n            input: {\n                shippingMethod: $shippingMethod\n            }) {\n            errors {\n                field\n                message\n                code\n            }\n            order {\n                shippingMethodName\n                shippingPrice {\n                net {\n                        amount\n                    }\n                    gross {\n                        amount\n                    }\n                tax {\n                    amount\n                    }\n                    net {\n                        amount\n                    }\n                    gross {\n                        amount\n                    }\n                }\n            shippingTaxRate\n            userEmail\n            }\n        }\n    }\n'

def test_draft_order_update_shipping_method_from_different_channel(staff_api_client, permission_group_manage_orders, draft_order, address_usa, shipping_method_channel_PLN):
    if False:
        i = 10
        return i + 15
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    order = draft_order
    order.shipping_address = address_usa
    order.save(update_fields=['shipping_address'])
    query = DRAFT_ORDER_UPDATE_SHIPPING_METHOD_MUTATION
    order_id = graphene.Node.to_global_id('Order', order.id)
    shipping_method_id = graphene.Node.to_global_id('ShippingMethod', shipping_method_channel_PLN.id)
    variables = {'id': order_id, 'shippingMethod': shipping_method_id}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['draftOrderUpdate']
    assert len(data['errors']) == 1
    assert not data['order']
    error = data['errors'][0]
    assert error['code'] == OrderErrorCode.SHIPPING_METHOD_NOT_APPLICABLE.name
    assert error['field'] == 'shippingMethod'

def test_draft_order_update_shipping_method_prices_updates(staff_api_client, permission_group_manage_orders, draft_order, address_usa, shipping_method, shipping_method_weight_based):
    if False:
        print('Hello World!')
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    order = draft_order
    order.shipping_address = address_usa
    order.shipping_method = shipping_method
    order.save(update_fields=['shipping_address', 'shipping_method'])
    assert shipping_method.channel_listings.first().price_amount == 10
    method_2 = shipping_method_weight_based
    m2_channel_listing = method_2.channel_listings.first()
    m2_channel_listing.price_amount = 15
    m2_channel_listing.save(update_fields=['price_amount'])
    query = DRAFT_ORDER_UPDATE_SHIPPING_METHOD_MUTATION
    order_id = graphene.Node.to_global_id('Order', order.id)
    shipping_method_id = graphene.Node.to_global_id('ShippingMethod', method_2.id)
    variables = {'id': order_id, 'shippingMethod': shipping_method_id}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['draftOrderUpdate']
    order.refresh_from_db()
    assert not data['errors']
    assert data['order']['shippingMethodName'] == method_2.name
    assert data['order']['shippingPrice']['net']['amount'] == 15.0

def test_draft_order_update_shipping_method_clear_with_none(staff_api_client, permission_group_manage_orders, draft_order, address_usa, shipping_method):
    if False:
        print('Hello World!')
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    order = draft_order
    order.shipping_address = address_usa
    order.shipping_method = shipping_method
    order.save(update_fields=['shipping_address', 'shipping_method'])
    query = DRAFT_ORDER_UPDATE_SHIPPING_METHOD_MUTATION
    order_id = graphene.Node.to_global_id('Order', order.id)
    variables = {'id': order_id, 'shippingMethod': None}
    zero_shipping_price_data = {'tax': {'amount': 0.0}, 'net': {'amount': 0.0}, 'gross': {'amount': 0.0}}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['draftOrderUpdate']
    order.refresh_from_db()
    assert not data['errors']
    assert data['order']['shippingMethodName'] is None
    assert data['order']['shippingPrice'] == zero_shipping_price_data
    assert data['order']['shippingTaxRate'] == 0.0
    assert order.shipping_method is None

def test_draft_order_update_shipping_method(staff_api_client, permission_group_manage_orders, draft_order, shipping_method):
    if False:
        while True:
            i = 10
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    order = draft_order
    order.shipping_method = None
    order.base_shipping_price = zero_money(order.currency)
    order.save()
    query = DRAFT_ORDER_UPDATE_SHIPPING_METHOD_MUTATION
    order_id = graphene.Node.to_global_id('Order', order.id)
    method_id = graphene.Node.to_global_id('ShippingMethod', shipping_method.id)
    variables = {'id': order_id, 'shippingMethod': method_id}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    order.refresh_from_db()
    shipping_total = shipping_method.channel_listings.get(channel_id=order.channel_id).get_total()
    shipping_price = TaxedMoney(shipping_total, shipping_total)
    data = content['data']['draftOrderUpdate']
    assert not data['errors']
    assert data['order']['shippingMethodName'] == shipping_method.name
    assert data['order']['shippingPrice']['net']['amount'] == quantize_price(shipping_price.net.amount, shipping_price.currency)
    assert data['order']['shippingPrice']['gross']['amount'] == quantize_price(shipping_price.gross.amount, shipping_price.currency)
    assert order.base_shipping_price == shipping_total
    assert order.shipping_method == shipping_method
    assert order.base_shipping_price == shipping_total
    assert order.shipping_price_net == shipping_price.net
    assert order.shipping_price_gross == shipping_price.gross

def test_draft_order_update_no_shipping_method_channel_listings(staff_api_client, permission_group_manage_orders, draft_order, shipping_method):
    if False:
        print('Hello World!')
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    order = draft_order
    order.shipping_method = None
    order.base_shipping_price = zero_money(order.currency)
    order.save()
    shipping_method.channel_listings.all().delete()
    query = DRAFT_ORDER_UPDATE_SHIPPING_METHOD_MUTATION
    order_id = graphene.Node.to_global_id('Order', order.id)
    method_id = graphene.Node.to_global_id('ShippingMethod', shipping_method.id)
    variables = {'id': order_id, 'shippingMethod': method_id}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['draftOrderUpdate']
    errors = data['errors']
    assert len(errors) == 1
    assert errors[0]['code'] == OrderErrorCode.SHIPPING_METHOD_NOT_APPLICABLE.name
    assert errors[0]['field'] == 'shippingMethod'