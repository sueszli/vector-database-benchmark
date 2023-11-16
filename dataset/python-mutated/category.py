from kams_erp.configs.install.configuration import InstallKamsERP_Configuration
from kams_erp.models.kqs_category import KqsKategorie

class InstallKamsERP_Category(InstallKamsERP_Configuration):

    def insert_or_find_category(self, category):
        if False:
            i = 10
            return i + 15
        '\n\n        :param category:\n        :return:\n        '
        odoo_category = self.xml_operand.find_category([[['kqs_original_id', '=', category.numer]]])
        if not odoo_category:
            if int(category.kat_matka) != 0:
                parent_kqs = self.session.query(KqsKategorie).filter(KqsKategorie.numer == category.kat_matka).first()
                parent_odoo_id = self.xml_operand.find_category([[['name', '=', parent_kqs.nazwa]]])
                if not parent_odoo_id:
                    category_to_insert = [{'name': parent_kqs.nazwa, 'kqs_original_id': parent_kqs.numer}]
                    parent_odoo_id = self.xml_operand.insert_category(category_to_insert)
                category_to_insert = [{'name': category.nazwa, 'parent_id': parent_odoo_id.get('id'), 'kqs_original_id': category.numer}]
                cat_id = self.xml_operand.insert_category(category_to_insert).get('id')
            else:
                category_to_insert = [{'name': category.nazwa, 'kqs_original_id': category.numer}]
                cat_id = self.xml_operand.insert_category(category_to_insert).get('id')
        else:
            cat_id = odoo_category.get('id')
        return cat_id

    def update_category(self, id_category, inserted_product):
        if False:
            return 10
        '\n\n        :param id_category:\n        :param inserted_product:\n        :return:\n        '
        data_to_update = {'product_ids': [(4, inserted_product.get('id'))]}
        self.xml_operand.update_category(id_category, data_to_update)