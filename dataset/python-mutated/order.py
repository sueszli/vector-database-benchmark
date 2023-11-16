import time
import datetime
from datetime import date
from kams_erp.configs.install.configuration import InstallKamsERP_Configuration
from kams_erp.configs.install.customer import InstallKamsERP_Customer
from kams_erp.configs.install.product import InstallKamsERP_Product
from kams_erp.models.kamserp_config import SALES_PERSON
from kams_erp.models.kqs_order import KqsZamowienia, KqsZamowieniaProdukty
from kams_erp.utils.xml_rpc_operations import convert_decimal_to_float
from odoo.tools import DEFAULT_SERVER_DATETIME_FORMAT

class InstallKamsERP_Order(InstallKamsERP_Configuration):

    def get_orders(self):
        if False:
            print('Hello World!')
        ' Gets orders from KQS and insert to Odoo '
        orders = self.session.query(KqsZamowienia).filter(KqsZamowienia.data >= self.__get_time_before(months=6)).filter(KqsZamowienia.id == 10651)
        salesperson = self.connector.search('res.users', [[['login', '=', SALES_PERSON]]])[0]
        for order in orders:
            customer = InstallKamsERP_Customer().create_or_update_customer(order)
            customer_order = self.__create_order(order, salesperson, customer)
            if customer_order:
                self.__create_order_line(order, customer_order)
            break

    def __create_order(self, order, salesperson, customer):
        if False:
            for i in range(10):
                print('nop')
        customer_order = self.xml_operand.find_order([[['unique_number', '=', order.unikalny_numer]]])
        if not customer_order:
            order_to_insert = {'name': order.id, 'user_id': salesperson, 'partner_id': customer.get('id'), 'state': 'draft', 'date_order': datetime.datetime.fromtimestamp(int(order.data)).strftime(DEFAULT_SERVER_DATETIME_FORMAT), 'create_date': datetime.datetime.fromtimestamp(int(order.data)).strftime(DEFAULT_SERVER_DATETIME_FORMAT), 'note': order.uwagi, 'unique_number': order.unikalny_numer, 'customer_ip': order.klient_ip, 'document_type': self.__get_proper_recipt(order)}
            if order.klient_nip != '':
                order_to_insert['partner_invoice_id'] = customer.get('parent_id')[0]
            if order.dokument > 0:
                order_to_insert['document_type'] = 'invoice'
            else:
                order_to_insert['document_type'] = 'receipt'
            customer_order = self.xml_operand.insert_order([order_to_insert])
        return customer_order

    def __create_order_line(self, order, customer_order):
        if False:
            while True:
                i = 10
        fetched_order_line = self.session.query(KqsZamowieniaProdukty).filter(KqsZamowieniaProdukty.zamowienie_id == order.id)
        for line in fetched_order_line:
            if line:
                order_line_to_insert = {'order_id': customer_order.get('id'), 'price_unit': float(InstallKamsERP_Product.get_netto_price_for_product(line)), 'price_tax': convert_decimal_to_float(line.cena * line.podatek / 100), 'qty_to_invoice': convert_decimal_to_float(line.ilosc), 'discount': convert_decimal_to_float(line.rabat), 'product_id': InstallKamsERP_Product.get_product_id(line)}
                self.xml_operand.insert_order_line([order_line_to_insert])
        shipment_to_insert = {'order_id': customer_order.get('id'), 'price_unit': float(InstallKamsERP_Product.calculate_netto_price(order.przesylka_koszt_brutto, 23)), 'product_id': self.xml_operand.find_product([[['name', '=', order.przesylka_nazwa]]]).get('id'), 'qty_to_invoice': 1}
        self.xml_operand.insert_order_line([shipment_to_insert])

    @staticmethod
    def __get_time_before(years=0, months=0, days=0):
        if False:
            for i in range(10):
                print('nop')
        year = date.today().year - years
        month = date.today().month - months
        if month <= 0:
            month += 12
            year -= 1
        day = date.today().day - days
        if day <= 0:
            day += 28
            month -= 1
        t = datetime.datetime(year, month, day, 0, 0)
        return time.mktime(t.timetuple())

    @staticmethod
    def __get_proper_recipt(order):
        if False:
            print('Hello World!')
        if order.dokument > 0:
            document_type = 'invoice'
        else:
            document_type = 'receipt'
        return document_type