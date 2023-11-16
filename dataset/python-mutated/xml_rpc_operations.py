import datetime
from decimal import Decimal
import time
from kams_erp.utils.xml_rpc_connector import XmlRpcConnector

class XmlRpcOperations(XmlRpcConnector):
    """
    Class responsible for operation with Odoo database.
    """

    def insert_products(self, product):
        if False:
            while True:
                i = 10
        '\n        Inserts new product record to Odoo database.\n        :param product: product to insert.\n        :return: Inserted record.\n        '
        created = self.create('product.product', product)
        return self.read('product.product', [[created]])[0]

    def find_product(self, param):
        if False:
            while True:
                i = 10
        '\n        Finds the product for given condition.\n        :param param: condition.\n        :return: product id.\n        '
        product_id = self.search('product.product', param)
        if len(product_id) > 0:
            product_id = product_id[0]
        return self.read('product.product', [product_id])

    def update_product(self, id_record, data_to_update, read=True):
        if False:
            return 10
        '\n        Updates the product with given data.\n        :param id_record: record to update.\n        :param data_to_update: data to update\n        :param read: Parameter for better performance. If read equals False then omit return value with read value.\n        :return: category data.\n        '
        self.update('product.product', [[id_record], data_to_update])
        if read:
            return self.read('product.product', [id_record])

    def update_product_template(self, id_record, data_to_update, read=True):
        if False:
            return 10
        '\n        Updates the product template with given data.\n        :param id_record: record to update.\n        :param data_to_update: data to update\n        :param read: Parameter for better performance. If read equals False then omit return value with read value.\n        :return: category data.\n        '
        self.update('product.template', [[id_record], data_to_update])
        if read:
            return self.read('product.template', [id_record])

    def find_product_template(self, param):
        if False:
            for i in range(10):
                print('nop')
        '\n        Finds the product template for given condition.\n        :param param: condition.\n        :return: product template id.\n        '
        product_id = self.search('product.template', param)
        if len(product_id) > 0:
            product_id = product_id[0]
        return self.read('product.template', [product_id])

    def find_tax(self, param):
        if False:
            while True:
                i = 10
        '\n        Finds the tax for given condition.\n        :param param: condition.\n        :return: product template id.\n        '
        tax_id = self.search('account.tax', param)
        if len(tax_id) > 0:
            tax_id = tax_id[0]
        return self.read('account.tax', [tax_id])

    def insert_category(self, category):
        if False:
            print('Hello World!')
        '\n        Inserts new category record to Odoo database.\n        :param category: category to insert.\n        :return: Inserted record.\n        '
        created = self.create('product.category', category)
        return self.read('product.category', [[created]])[0]

    def find_category(self, param):
        if False:
            while True:
                i = 10
        '\n        Finds the category for given condition.\n        :param param: condition.\n        :return: category data.\n        '
        category_id = self.search('product.category', param)
        if len(category_id) > 0:
            category_id = category_id[0]
        category_obj = self.read('product.category', [category_id])
        if isinstance(category_obj, list):
            if len(category_obj) > 0:
                category_obj = category_obj[0]
        return category_obj

    def update_category(self, id_record, data_to_update, read=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Updates the category with given data\n        :param id_record: record to update.\n        :param data_to_update: data to update\n        :param read: Parameter for better performance. If read equals False then omit return value with read value.\n        :return: category data.\n        '
        self.update('product.category', [[id_record], data_to_update])
        if read:
            self.read('product.category', [id_record])

    def find_product_uom(self, param):
        if False:
            i = 10
            return i + 15
        '\n        Finds the product uom for given condition.\n        :param param: condition.\n        :return: product uom data.\n        '
        product_uom = self.search('product.uom', param)
        if len(product_uom) > 0:
            product_uom = product_uom[0]
        return self.read('product.uom', [product_uom])

    def update_product_uom(self, id_record, data_to_update, read=True):
        if False:
            print('Hello World!')
        '\n        Updates the product uom with given data\n        :param id_record: record to update.\n        :param data_to_update: data to update\n        :param read: Parameter for better performance. If read equals False then omit return value with read value.\n        :return: category data.\n        '
        self.update('product.uom', [[id_record], data_to_update])
        if read:
            self.read('product.uom', [id_record])

    def insert_partner(self, partner):
        if False:
            i = 10
            return i + 15
        '\n        Inserts new partner to Odoo database.\n        :param partner: Inserted record.\n        '
        created = self.create('res.partner', partner)
        return self.read('res.partner', [[created]])[0]

    def find_partner(self, param):
        if False:
            for i in range(10):
                print('nop')
        '\n        Finds the partner for given condition.\n        :param param: condition.\n        :return: manufacturer data.\n        '
        partner_id = self.search('res.partner', param)
        if len(partner_id) > 0:
            partner_id = partner_id[0]
        partner_obj = self.read('res.partner', [partner_id])
        if isinstance(partner_obj, list):
            if len(partner_obj) > 0:
                partner_obj = partner_obj[0]
        return partner_obj

    def insert_supplier(self, supplier):
        if False:
            for i in range(10):
                print('nop')
        '\n        Inserts new supplier to Odoo database.\n        :param supplier: Inserted record.\n        '
        created = self.create('product.supplierinfo', supplier)
        return self.read('product.supplierinfo', [[created]])[0]

    def find_supplier(self, param):
        if False:
            i = 10
            return i + 15
        '\n        Finds the supplier for given condition.\n        :param param: condition.\n        :return: supplier data.\n        '
        supplier_id = self.search('product.supplierinfo', param)
        if len(supplier_id) > 0:
            supplier_id = supplier_id[0]
        return self.read('product.supplierinfo', [supplier_id])

    def insert_product_seller(self, seller):
        if False:
            print('Hello World!')
        '\n        Inserts new seller record to Odoo database.\n        :param seller: seller to insert.\n        :return: Inserted record.\n        '
        created = self.create('product.supplierinfo', seller)
        return self.read('product.supplierinfo', [[created]])[0]

    def update_product_seller(self, id_record, data_to_update, read=True):
        if False:
            while True:
                i = 10
        '\n        Updates the product sellet with given data\n        :param id_record: record to update.\n        :param data_to_update: data to update\n        :param read: Parameter for better performance. If read equals False then omit return value with read value.\n        :return: stock quantity data.\n        '
        self.update('product.supplierinfo', [[id_record], data_to_update])
        if read:
            return self.read('product.supplierinfo', [id_record])

    def insert_attribute(self, attribute):
        if False:
            while True:
                i = 10
        '\n        Inserts new attribute record to Odoo database.\n        :param attribute: attribute to insert.\n        :return: Inserted record.\n        '
        created = self.create('product.attribute', attribute)
        return self.read('product.attribute', [[created]])[0]

    def find_attribute(self, param):
        if False:
            return 10
        '\n        Finds the attribute for given condition.\n        :param param: condition.\n        :return: attribute data.\n        '
        attribute_id = self.search('product.attribute', param)
        if len(attribute_id) > 0:
            attribute_id = attribute_id[0]
        attribute_obj = self.read('product.attribute', [attribute_id])
        if isinstance(attribute_obj, list):
            if len(attribute_obj) > 0:
                attribute_obj = attribute_obj[0]
        return attribute_obj

    def insert_attribute_values(self, attribute_value):
        if False:
            while True:
                i = 10
        '\n        Inserts new attribute values record to Odoo database.\n        :param attribute_value: attribute to insert.\n        :return: Inserted record.\n        '
        created = self.create('product.attribute.value', attribute_value)
        return self.read('product.attribute.value', [[created]])[0]

    def find_attribute_values(self, param):
        if False:
            for i in range(10):
                print('nop')
        '\n        Finds the attribute values for given condition.\n        :param param: condition.\n        :return: attribute values data.\n        '
        attribute_values_id = self.search('product.attribute.value', param)
        if len(attribute_values_id) > 0:
            attribute_values_id = attribute_values_id[0]
        attribute_vaules_obj = self.read('product.attribute.value', [attribute_values_id])
        if isinstance(attribute_vaules_obj, list):
            if len(attribute_vaules_obj) > 0:
                attribute_vaules_obj = attribute_vaules_obj[0]
        return attribute_vaules_obj

    def update_attribute_values(self, id_record, data_to_update, read=True):
        if False:
            i = 10
            return i + 15
        '\n        Updates the attribute values with given data\n        :param id_record: record to update.\n        :param data_to_update: data to update\n        :param read: Parameter for better performance. If read equals False then omit return value with read value.\n        :return: attribute line data.\n        '
        self.update('product.attribute.value', [[id_record], data_to_update])
        if read:
            return self.read('product.attribute.value', [id_record])

    def find_attribute_price(self, param):
        if False:
            for i in range(10):
                print('nop')
        '\n        Finds the attribute price for given condition.\n        :param param: condition.\n        :return: attribute line data.\n        '
        attribute_price_id = self.search('product.attribute.price', param)
        if len(attribute_price_id) > 0:
            attribute_price_id = attribute_price_id[0]
        return self.read('product.attribute.price', [attribute_price_id])

    def insert_attribute_price(self, attribute_price):
        if False:
            return 10
        '\n        Inserts new attribute values record to Odoo database.\n        :param attribute_price: attribute to insert.\n        :return: Inserted record.\n        '
        created = self.create('product.attribute.price', attribute_price)
        return self.read('product.attribute.price', [[created]])[0]

    def insert_attribute_line(self, attribute_line):
        if False:
            return 10
        '\n        Inserts new attribute line record to Odoo database.\n        :param attribute_line: attribute line to insert.\n        :return: Inserted record.\n        '
        created = self.create('product.attribute.line', attribute_line)
        return self.read('product.attribute.line', [[created]])[0]

    def update_attribute_line(self, id_record, data_to_update, read=True):
        if False:
            while True:
                i = 10
        '\n        Updates the attribute line with given data\n        :param id_record: record to update.\n        :param data_to_update: data to update\n        :param read: Parameter for better performance. If read equals False then omit return value with read value.\n        :return: attribute line data.\n        '
        self.update('product.attribute.line', [[id_record], data_to_update])
        if read:
            return self.read('product.attribute.line', [id_record])

    def find_attribute_line(self, param):
        if False:
            print('Hello World!')
        '\n        Finds the attribute line for given condition.\n        :param param: condition.\n        :return: attribute line data.\n        '
        attribute_line_id = self.search('product.attribute.line', param)
        if len(attribute_line_id) > 0:
            attribute_line_id = attribute_line_id[0]
        attribute_line_obj = self.read('product.attribute.line', [attribute_line_id])
        if isinstance(attribute_line_obj, list):
            if len(attribute_line_obj) > 0:
                attribute_line_obj = attribute_line_obj[0]
        return attribute_line_obj

    def insert_stock_move(self, stock_move):
        if False:
            i = 10
            return i + 15
        '\n        Inserts new stock move record to stock move to insert.\n        :param stock_move: stock move to insert.\n        :return: Inserted record.\n        '
        created = self.create('stock.move', stock_move)
        return self.read('stock.move', [[created]])[0]

    def find_stock_move(self, param):
        if False:
            for i in range(10):
                print('nop')
        '\n        Finds the stock quantity for given condition.\n        :param param: condition.\n        :return: stock quantity data.\n        '
        stock_quant_id = self.search('stock.move', param)
        if len(stock_quant_id) > 0:
            stock_quant_id = stock_quant_id[0]
        return self.read('stock.move', [stock_quant_id])

    def update_stock_move(self, id_record, data_to_update, read=True):
        if False:
            i = 10
            return i + 15
        '\n        Updates the stock quantity with given data\n        :param id_record: record to update.\n        :param data_to_update: data to update\n        :param read: Parameter for better performance. If read equals False then omit return value with read value.\n        :return: stock quantity data.\n        '
        self.update('stock.move', [[id_record], data_to_update])
        if read:
            return self.read('stock.move', [id_record])

    def insert_stock_quant(self, stock_quant):
        if False:
            return 10
        '\n        Inserts new stock quant record to stock quant to insert.\n        :param stock_quant: stock quant to insert.\n        :return: Inserted record.\n        '
        created = self.create('stock.quant', stock_quant)
        return self.read('stock.quant', [[created]])[0]

    def find_stock_quant(self, param):
        if False:
            i = 10
            return i + 15
        '\n        Finds the stock quantity for given condition.\n        :param param: condition.\n        :return: stock quantity data.\n        '
        stock_quant_id = self.search('stock.quant', param)
        if len(stock_quant_id) > 0:
            stock_quant_id = stock_quant_id[0]
        return self.read('stock.quant', [stock_quant_id])

    def update_stock_quant(self, id_record, data_to_update, read=True):
        if False:
            print('Hello World!')
        '\n        Updates the stock quantity with given data\n        :param id_record: record to update.\n        :param data_to_update: data to update\n        :param read: Parameter for better performance. If read equals False then omit return value with read value.\n        :return: stock quantity data.\n        '
        self.update('stock.quant', [[id_record], data_to_update])
        if read:
            return self.read('stock.quant', [id_record])

    def find_company(self, param):
        if False:
            i = 10
            return i + 15
        '\n        Finds the company for given condition.\n        :param param: condition.\n        :return: company data.\n        '
        comapny_id = self.search('res.company', param)
        if len(comapny_id) > 0:
            comapny_id = comapny_id[0]
        return self.read('res.company', [comapny_id])

    def insert_company(self, company):
        if False:
            return 10
        '\n        Inserts new comapny record.\n        :param company: company to insert.\n        :return: Inserted record.\n        '
        created = self.create('res.company', company)
        return self.read('res.company', [[created]])[0]

    def find_customer(self, param):
        if False:
            for i in range(10):
                print('nop')
        '\n        Finds the customer for given condition.\n        :param param: condition.\n        :return: customer data.\n        '
        customer_id = self.search('res.partner', param)
        if len(customer_id) > 0:
            customer_id = customer_id[0]
        return self.read('res.partner', [customer_id])

    def insert_customer(self, customer):
        if False:
            while True:
                i = 10
        '\n        Inserts new customer record.\n        :param customer: company to insert.\n        :return: Inserted record.\n        '
        created = self.create('res.partner', customer)
        return self.read('res.partner', [[created]])[0]

    def find_order(self, param):
        if False:
            i = 10
            return i + 15
        '\n        Finds the order for given condition.\n        :param param: condition.\n        :return: order data.\n        '
        sale_order_id = self.search('sale.order', param)
        if len(sale_order_id) > 0:
            sale_order_id = sale_order_id[0]
        return self.read('sale.order', [sale_order_id])

    def insert_order(self, order):
        if False:
            for i in range(10):
                print('nop')
        '\n        Inserts new order record.\n        :param order: company to insert.\n        :return: Inserted record.\n        '
        created = self.create('sale.order', order)
        return self.read('sale.order', [[created]])[0]

    def insert_order_line(self, order_line):
        if False:
            while True:
                i = 10
        '\n        Inserts new order record.\n        :param order_line: company to insert.\n        :return: Inserted record.\n        '
        created = self.create('sale.order.line', order_line)
        return self.read('sale.order.line', [[created]])[0]

def _datetime_to_integer(dt_time, template):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert datetime to intiger.\n    :param template: example "%Y-%m-%d %H:%M:%S.%f"\n    :param dt_time: datatime object.\n    :return: int value of datatime.\n    '
    if type(dt_time) is datetime.datetime:
        dt_time = str(dt_time)
    return time.mktime(datetime.datetime.strptime(dt_time, template).timetuple())

def convert_decimal_to_float(ob):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert decimal value to float, avoid marshal error.\n    :param ob: decimal object.\n    :return: float value.\n    '
    if isinstance(ob, Decimal):
        return float(ob)
    if isinstance(ob, (tuple, list)):
        return [convert_decimal_to_float(v) for v in ob]
    if isinstance(ob, dict):
        return {k: convert_decimal_to_float(v) for (k, v) in ob.iteritems()}
    return ob