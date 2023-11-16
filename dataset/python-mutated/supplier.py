import base64
import urllib2
from kams_erp.configs.install.configuration import InstallKamsERP_Configuration
from kams_erp.configs.install.manufacturer import InstallKamsERP_Manufacturer

class InstallKamsERP_Supplier(InstallKamsERP_Configuration):

    def insert_or_find_supplier(self, suppliers, producent_id, purchase_price):
        if False:
            i = 10
            return i + 15
        '\n\n        :param suppliers:\n        :param producent_id:\n        :param purchase_price:\n        :return:\n        '
        if producent_id == 0:
            odoo_supplier = self.xml_operand.find_partner([[['kqs_original_id', '=', producent_id]]])
            if not odoo_supplier:
                odoo_supplier_to_insert = [{'name': 'Unknown', 'kqs_original_id': producent_id, 'supplier': True}]
                supplier_id = self.xml_operand.insert_partner(odoo_supplier_to_insert).get('id')
            else:
                supplier_id = odoo_supplier.get('id')
        else:
            supplier_to_insert = [{'name': InstallKamsERP_Manufacturer().insert_or_find_manufacturer(suppliers, producent_id, True), 'kqs_original_id': producent_id, 'price': purchase_price, 'image': base64.encodestring(urllib2.urlopen(self.url + '/base/static/src/img/company_image.png').read())}]
            supplier_id = self.xml_operand.insert_supplier(supplier_to_insert).get('id')
        return supplier_id