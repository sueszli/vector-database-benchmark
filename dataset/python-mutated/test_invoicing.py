from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, patch
import pytz
from prices import Money
from ....giftcard.events import gift_cards_used_in_order_event
from ....giftcard.models import GiftCard
from ..utils import chunk_products, generate_invoice_number, generate_invoice_pdf, get_gift_cards_payment_amount, get_product_limit_first_page, make_full_invoice_number

def test_chunk_products(product):
    if False:
        return 10
    assert chunk_products([product] * 3, 3) == [[product] * 3]
    assert chunk_products([product] * 5, 3) == [[product] * 3, [product] * 2]
    assert chunk_products([product] * 8, 3) == [[product] * 3, [product] * 3, [product] * 2]

def test_get_product_limit_first_page(product):
    if False:
        i = 10
        return i + 15
    assert get_product_limit_first_page([product] * 3) == 3
    assert get_product_limit_first_page([product] * 4) == 4
    assert get_product_limit_first_page([product] * 16) == 4

@patch('saleor.plugins.invoicing.utils.HTML')
@patch('saleor.plugins.invoicing.utils.get_template')
@patch('saleor.plugins.invoicing.utils.os')
def test_generate_invoice_pdf_for_order(os_mock, get_template_mock, HTML_mock, fulfilled_order, customer_user, gift_card):
    if False:
        return 10
    get_template_mock.return_value.render = Mock(return_value='<html></html>')
    os_mock.path.join.return_value = 'test'
    previous_current_balance = gift_card.current_balance
    gift_card.current_balance = Money(Decimal(5.0), 'USD')
    gift_card.save(update_fields=['current_balance_amount'])
    balance_data = [(gift_card, previous_current_balance.amount)]
    gift_cards_used_in_order_event(balance_data, fulfilled_order, customer_user, None)
    (content, creation) = generate_invoice_pdf(fulfilled_order.invoices.first())
    get_template_mock.return_value.render.assert_called_once_with({'invoice': fulfilled_order.invoices.first(), 'creation_date': datetime.now(tz=pytz.utc).strftime('%d %b %Y'), 'order': fulfilled_order, 'gift_cards_payment': previous_current_balance - gift_card.current_balance, 'font_path': 'file://test', 'products_first_page': list(fulfilled_order.lines.all()), 'rest_of_products': []})
    HTML_mock.assert_called_once_with(string=get_template_mock.return_value.render.return_value)

def test_generate_invoice_number_invalid_numeration(fulfilled_order):
    if False:
        while True:
            i = 10
    invoice = fulfilled_order.invoices.last()
    invoice.number = 'invalid/06/2020'
    invoice.save(update_fields=['number'])
    assert generate_invoice_number() == make_full_invoice_number()

def test_generate_invoice_number_no_existing_invoice(fulfilled_order):
    if False:
        for i in range(10):
            print('nop')
    fulfilled_order.invoices.all().delete()
    assert generate_invoice_number() == make_full_invoice_number()

def test_get_gift_cards_payment_amount(order, gift_card, gift_card_expiry_date, gift_card_used, customer_user):
    if False:
        return 10
    previous_current_balance_gift_card = gift_card.current_balance.amount
    previous_current_balance_gift_card_used = gift_card_used.current_balance.amount
    new_current_value = Decimal(5.0)
    gift_card.current_balance = Money(new_current_value, 'USD')
    gift_card_used.current_balance = Money(new_current_value, 'USD')
    GiftCard.objects.bulk_update([gift_card, gift_card_used], ['current_balance_amount'])
    balance_data = [(gift_card, previous_current_balance_gift_card), (gift_card_used, previous_current_balance_gift_card_used)]
    gift_cards_used_in_order_event(balance_data, order, customer_user, None)
    gift_cards_payment = get_gift_cards_payment_amount(order)
    value = previous_current_balance_gift_card - new_current_value + (previous_current_balance_gift_card_used - new_current_value)
    assert gift_cards_payment == Money(value, order.currency)

def test_get_gift_cards_payment_amount_equal_zero(order, gift_card, gift_card_expiry_date, gift_card_used, customer_user):
    if False:
        while True:
            i = 10
    gift_cards_payment = get_gift_cards_payment_amount(order)
    assert gift_cards_payment == Money(0, order.currency)