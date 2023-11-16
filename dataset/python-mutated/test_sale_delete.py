from unittest.mock import patch
import graphene
import pytest
from .....discount.error_codes import DiscountErrorCode
from .....discount.models import Promotion, PromotionRule
from ....tests.utils import get_graphql_content
from ...utils import convert_migrated_sale_predicate_to_catalogue_info
SALE_DELETE_MUTATION = '\n    mutation DeleteSale($id: ID!) {\n        saleDelete(id: $id) {\n            sale {\n                name\n                id\n            }\n            errors {\n                field\n                code\n                message\n            }\n            }\n        }\n'

@patch('saleor.product.tasks.update_products_discounted_prices_for_promotion_task.delay')
@patch('saleor.plugins.manager.PluginsManager.sale_deleted')
def test_sale_delete_mutation(deleted_webhook_mock, update_products_discounted_prices_for_promotion_task_mock, staff_api_client, promotion_converted_from_sale, catalogue_predicate, permission_manage_discounts):
    if False:
        i = 10
        return i + 15
    query = SALE_DELETE_MUTATION
    promotion = promotion_converted_from_sale
    previous_catalogue = convert_migrated_sale_predicate_to_catalogue_info(catalogue_predicate)
    variables = {'id': graphene.Node.to_global_id('Sale', promotion.old_sale_id)}
    response = staff_api_client.post_graphql(query, variables, permissions=[permission_manage_discounts])
    content = get_graphql_content(response)
    assert not content['data']['saleDelete']['errors']
    data = content['data']['saleDelete']['sale']
    assert data['name'] == promotion.name
    assert data['id'] == graphene.Node.to_global_id('Sale', promotion.old_sale_id)
    assert not Promotion.objects.filter(id=promotion.id).first()
    assert not PromotionRule.objects.filter(promotion_id=promotion.id).first()
    with pytest.raises(promotion._meta.model.DoesNotExist):
        promotion.refresh_from_db()
    deleted_webhook_mock.assert_called_once_with(promotion, previous_catalogue)
    update_products_discounted_prices_for_promotion_task_mock.assert_called_once()

@patch('saleor.product.tasks.update_products_discounted_prices_for_promotion_task.delay')
@patch('saleor.plugins.manager.PluginsManager.sale_deleted')
def test_sale_delete_mutation_with_promotion_id(deleted_webhook_mock, update_products_discounted_prices_for_promotion_task_mock, staff_api_client, promotion_converted_from_sale, permission_manage_discounts):
    if False:
        return 10
    query = SALE_DELETE_MUTATION
    promotion = promotion_converted_from_sale
    variables = {'id': graphene.Node.to_global_id('Promotion', promotion.id)}
    response = staff_api_client.post_graphql(query, variables, permissions=[permission_manage_discounts])
    content = get_graphql_content(response)
    assert not content['data']['saleDelete']['sale']
    errors = content['data']['saleDelete']['errors']
    assert len(errors) == 1
    assert errors[0]['field'] == 'id'
    assert errors[0]['code'] == DiscountErrorCode.INVALID.name
    assert errors[0]['message'] == "Provided ID refers to Promotion model. Please use 'promotionDelete' mutation instead."
    deleted_webhook_mock.assert_not_called()
    update_products_discounted_prices_for_promotion_task_mock.assert_not_called()

def test_sale_delete_not_found_error(staff_api_client, permission_manage_discounts):
    if False:
        for i in range(10):
            print('nop')
    query = SALE_DELETE_MUTATION
    variables = {'id': graphene.Node.to_global_id('Sale', '0')}
    response = staff_api_client.post_graphql(query, variables, permissions=[permission_manage_discounts])
    content = get_graphql_content(response)
    assert not content['data']['saleDelete']['sale']
    errors = content['data']['saleDelete']['errors']
    assert len(errors) == 1
    assert errors[0]['field'] == 'id'
    assert errors[0]['code'] == DiscountErrorCode.NOT_FOUND.name