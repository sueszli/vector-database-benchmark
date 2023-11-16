# -*- coding: utf-8 -*-
# @COPYRIGHT_begin
#
# Copyright [2015] Michał Szczygieł (m4gik), M4GiK Software
#
# @COPYRIGHT_end

import base64
import urllib2

from functools import partial
from kams_erp.configs.install.category import InstallKamsERP_Category
from kams_erp.configs.install.configuration import InstallKamsERP_Configuration
from kams_erp.configs.install.manufacturer import InstallKamsERP_Manufacturer
from kams_erp.configs.install.supplier import InstallKamsERP_Supplier
from kams_erp.models.kamserp_config import DOMAIN, ODOO_DATABASE_USER, ODOO_DATABASE_PASSWORD, SHIPMENT_PREPAYMENT_NAME, \
    SHIPMENT_PREPAYMENT_DESCRIPTION, SHIPMENT_PREPAYMENT_PRICE, SHIPMENT_PREPAYMENT_COST, \
    SHIPMENT_PERSONAL_COLLECTION_NAME, SHIPMENT_PERSONAL_COLLECTION_DESCRIPTION, SHIPMENT_PERSONAL_COLLECTION_PRICE, \
    SHIPMENT_PERSONAL_COLLECTION_COST, SHIPMENT_PAYMENT_ON_DELIVERY_NAME, SHIPMENT_PAYMENT_ON_DELIVERY_DESCRIPTION, \
    SHIPMENT_PAYMENT_ON_DELIVERY_PRICE, SHIPMENT_PAYMENT_ON_DELIVERY_COST, SHIPMENT_INPOST_NAME, \
    SHIPMENT_INPOST_DESCRIPTION, SHIPMENT_INPOST_PRICE, SHIPMENT_INPOST_COST
from kams_erp.models.kqs_images import KqsGaleriaZaczepy, KqsGaleria
from kams_erp.models.kqs_products_attribute import KqsProduktyOpcje, KqsProduktyWartosci
from kams_erp.models.subiekt_product import get_last_buy_price
from kams_erp.utils.kams_erp_tools import clean_html
from kams_erp.utils.xml_rpc_operations import convert_decimal_to_float


class InstallKamsERP_Product(InstallKamsERP_Configuration):
    def insert_product(self, product, category, manufacturers, suppliers, attributes, price=0.00, stan=0):
        """

        :param product:
        :param category:
        :param manufacturers:
        :param suppliers:
        :param attributes:
        :param price:
        :param stan:
        :return:
        """
        purchase_price = convert_decimal_to_float(
            get_last_buy_price(self.subiekt_session, product.kod_kreskowy))

        id_category = InstallKamsERP_Category().insert_or_find_category(category)
        id_manufacturer = InstallKamsERP_Manufacturer().insert_or_find_manufacturer(manufacturers, product.producent_id,
                                                                                    False)
        id_supplier = InstallKamsERP_Supplier().insert_or_find_supplier(suppliers, product.dostawca_id, purchase_price)

        # Prepare object to insert
        product_template = {
            'name': product.nazwa,
            'description': product.krotki_opis + '\n' + clean_html(product.opis),
            'price': float(self.get_netto_price_for_product(product)),
            'price_subiekt': float(price),
            'description_sale': product.krotki_opis,
            # 'warehouse_id':
            #     self.connector.search('stock.warehouse', [[['name', '=', 'Kams Magazyn']]])[0],
            'manufacturer_id': id_manufacturer,
            'seller_ids': [(4, id_supplier)],
            'weight': convert_decimal_to_float(product.waga),
            'categ_id': id_category,
            'standard_price': purchase_price,
            'taxes_id': [(6, 0, [self.__get_tax_id(product)])],
        }
        image_name = self.__get_image(product.numer)
        if image_name:
            image = DOMAIN + "/galerie/" + image_name.obraz[0] + "/" + image_name.obraz + ".jpg"
            # image_small = DOMAIN + "/galerie/" + image_name.obraz[0] + "/" + image_name.obraz + "_k.jpg"
            # image_medium = DOMAIN + "/galerie/" + image_name.obraz[0] + "/" + image_name.obraz + "_m.jpg"
            product_template['image'] = base64.encodestring(urllib2.urlopen(image).read())
            # product_template['image_medium'] = base64.encodestring(urllib2.urlopen(image_medium).read())
            # product_template['image_small'] = base64.encodestring(urllib2.urlopen(image_small).read())

        if product.kod_kreskowy != '':
            product_template['barcode'] = str(product.kod_kreskowy)
        print product.nazwa
        inserted_product = self.xml_operand.insert_products([product_template])
        self.__insert_kqs_product_number(product, inserted_product)
        self.__insert_or_find_attributes(attributes, product, inserted_product)
        self.__update_products_variant_price(inserted_product, purchase_price, price)
        self.__update_product_quantity(inserted_product, convert_decimal_to_float(stan))

    def insert_transport_as_product(self):
        """

        :return:
        """
        if not self.xml_operand.find_product([[['name', '=', SHIPMENT_PREPAYMENT_NAME]]]):
            self.xml_operand.insert_products([self.__get_shipment_xml(SHIPMENT_PREPAYMENT_NAME,
                                                                      SHIPMENT_PREPAYMENT_DESCRIPTION,
                                                                      SHIPMENT_PREPAYMENT_PRICE,
                                                                      SHIPMENT_PREPAYMENT_COST)])

        if not self.xml_operand.find_product([[['name', '=', SHIPMENT_PERSONAL_COLLECTION_NAME]]]):
            self.xml_operand.insert_products([self.__get_shipment_xml(SHIPMENT_PERSONAL_COLLECTION_NAME,
                                                                      SHIPMENT_PERSONAL_COLLECTION_DESCRIPTION,
                                                                      SHIPMENT_PERSONAL_COLLECTION_PRICE,
                                                                      SHIPMENT_PERSONAL_COLLECTION_COST)])

        if not self.xml_operand.find_product([[['name', '=', SHIPMENT_PAYMENT_ON_DELIVERY_NAME]]]):
            self.xml_operand.insert_products([self.__get_shipment_xml(SHIPMENT_PAYMENT_ON_DELIVERY_NAME,
                                                                      SHIPMENT_PAYMENT_ON_DELIVERY_DESCRIPTION,
                                                                      SHIPMENT_PAYMENT_ON_DELIVERY_PRICE,
                                                                      SHIPMENT_PAYMENT_ON_DELIVERY_COST)])

        if not self.xml_operand.find_product([[['name', '=', SHIPMENT_INPOST_NAME]]]):
            self.xml_operand.insert_products([self.__get_shipment_xml(SHIPMENT_INPOST_NAME,
                                                                      SHIPMENT_INPOST_DESCRIPTION,
                                                                      SHIPMENT_INPOST_PRICE,
                                                                      SHIPMENT_INPOST_COST)])

    def __get_image_with_credentials(self, image_path):
        request = urllib2.Request(self.url + image_path)
        base64string = base64.b64encode('%s:%s' % (ODOO_DATABASE_USER, ODOO_DATABASE_PASSWORD))
        request.add_header("Authorization", "Basic %s" % base64string)
        return urllib2.urlopen(request).read()

    def __get_tax_id(self, product):
        if product.podatek == 8:
            tax = self.xml_operand.find_tax([[['name', '=', 'VAT-8%']]])[0]
        else:
            tax = self.xml_operand.find_tax([[['name', '=', 'VAT-23%']]])[0]

        return tax.get('id')

    def __get_image(self, product_id):
        query = self.session.query(KqsGaleriaZaczepy).filter(
            KqsGaleriaZaczepy.produkt_id == product_id).filter(
            KqsGaleriaZaczepy.kolejnosc == 1)
        image = query.first()
        if image:
            get_image = self.session.query(KqsGaleria).filter(KqsGaleria.numer == image.obraz_id).first()
        else:
            get_image = None
        return get_image

    def __insert_kqs_product_number(self, product, inserted_product):
        unique_number_to_insert = {
            'unique_product_number': product.kod_produktu,
        }
        self.xml_operand.update_product_template(inserted_product.get('product_tmpl_id')[0],
                                                 unique_number_to_insert, read=False)

    def __insert_or_find_attributes(self, attributes, product, inserted_product):
        attributes_lst = filter(partial(self.get_product_attributes, product=product), attributes)
        values_to_insert = []
        attribute_kind = None
        for attribute in attributes_lst:
            options = self.session.query(KqsProduktyOpcje).filter(
                KqsProduktyOpcje.numer == attribute.opcja_id).all()
            for option in options:
                odoo_attribute = self.xml_operand.find_attribute(
                    [[['kqs_original_id', '=', option.numer]]])
                if not odoo_attribute:
                    attribute_to_insert = [{
                        'name': option.opcja,
                        'sequence': option.kolejnosc,
                        'kqs_original_id': option.numer,
                    }]
                    odoo_attribute = self.xml_operand.insert_attribute(attribute_to_insert)
                if attribute_kind != odoo_attribute.get('id'):
                    values_to_insert = []
                attribute_kind = odoo_attribute.get('id')
                values = self.session.query(KqsProduktyWartosci).filter(
                    KqsProduktyWartosci.numer == attribute.wartosc_id).all()
                for value in values:
                    odoo_attribute_value = self.xml_operand.find_attribute_values(
                        [['&', ['name', '=', value.wartosc], ['attribute_id', 'ilike',
                                                              odoo_attribute.get('id')]]])
                    if not odoo_attribute_value:
                        attribute_value_to_insert = [{
                            'name': value.wartosc,
                            'price_extra': float(value.zmiana_wartosc),
                            'attribute_id': odoo_attribute.get('id'),
                            'kqs_original_id': value.numer,
                            'sequence': value.kolejnosc,
                        }]
                        odoo_attribute_value = self.xml_operand.insert_attribute_values(attribute_value_to_insert)

                    values_to_insert.append(odoo_attribute_value.get('id'))
                    odoo_attribute_line = self.xml_operand.find_attribute_line([['&', ['product_tmpl_id', '=',
                                                                                       inserted_product.get(
                                                                                           'product_tmpl_id')[0]],
                                                                                 ['attribute_id', 'ilike',
                                                                                  odoo_attribute.get('id')]]])

                    attribute_line_to_insert = {
                        'product_tmpl_id': inserted_product.get('product_tmpl_id')[0],
                        'attribute_id': odoo_attribute.get('id'),
                        'value_ids': [(6, 0, values_to_insert)]
                    }
                    if not odoo_attribute_line:
                        odoo_attribute_line = self.xml_operand.insert_attribute_line([attribute_line_to_insert])
                    else:
                        self.xml_operand.update_attribute_line(odoo_attribute_line.get('id'),
                                                               attribute_line_to_insert, read=False)
                    product_variants_to_update = {
                        'attribute_line_ids': [(4, odoo_attribute_line.get('id'))],
                    }

                    self.xml_operand.update_product_template(inserted_product.get('product_tmpl_id')[0],
                                                             product_variants_to_update, read=False)

    def __update_products_variant_price(self, inserted_product, purchase_price, subiekt_price):
        product_list_ids = self.connector.search('product.product', [
            [['product_tmpl_id', '=', inserted_product.get('product_tmpl_id')[0]]]])

        for product_id in product_list_ids:
            product_to_update = {
                'standard_price': purchase_price,
                'price_subiekt': float(subiekt_price),
            }
            self.xml_operand.update_product(product_id, product_to_update, read=False)

    def __update_product_quantity(self, inserted_product, product_quantity):

        stock_quant = self.xml_operand.find_stock_quant([[['product_id', '=', inserted_product.get('id')]]])

        if not stock_quant:
            stock_to_insert = [{
                'name': "Aktualizacja Stanu",
                'product_id': self.connector.search('product.product', [
                    [['product_tmpl_id', '=', inserted_product.get('product_tmpl_id')[0]]]])[0],
                'qty': product_quantity,
                'location_id': self.connector.search('stock.location', [[['name', '=', 'WH']]])[0],
            }]
            self.xml_operand.insert_stock_quant(stock_to_insert)
        else:
            stock_to_update = {
                'qty': product_quantity
            }
            self.xml_operand.update_stock_quant(stock_quant.get('id'), stock_to_update, read=False)

    def __get_shipment_xml(self, shipment_name, shipment_description, shipment_price, shipment_cost):
        odoo_category = self.xml_operand.find_category([[['name', '=', 'Transport']]])
        if not odoo_category:
            category_to_insert = [{
                'name': 'Transport',
            }]
            odoo_category = self.xml_operand.insert_category(category_to_insert)

        return {
            'name': shipment_name,
            'categ_id': odoo_category.get('id'),
            'description': shipment_description,
            'description_sale': shipment_description,
            'price': float(InstallKamsERP_Product.calculate_netto_price(shipment_price, 23)),
            'price_subiekt': float(InstallKamsERP_Product.calculate_netto_price(shipment_price, 23)),
            'standard_price': float(InstallKamsERP_Product.calculate_netto_price(shipment_cost, 23)),
            'image': base64.encodestring(self.__get_image_with_credentials('/base/static/src/img/truck.png')),
            # 'warehouse_id':
            #     self.connector.search('stock.warehouse', [[['name', '=', 'Kams Magazyn']]])[0],
        }

    @staticmethod
    def calculate_netto_price(price, tax):
        """

        :param price:
        :param tax:
        :return:
        """
        return convert_decimal_to_float(price) - (
            (convert_decimal_to_float(price) * (tax / 100.0)) / ((tax / 100.0) + 1.00))

    @staticmethod
    def get_netto_price_for_product(product):
        """

        :param product:
        :return:
        """
        price = 0.00
        if product.podatek == 8:
            price = InstallKamsERP_Product.calculate_netto_price(product.cena, 8)

        if product.podatek == 23:
            price = InstallKamsERP_Product.calculate_netto_price(product.cena, 23)

        return price

    @staticmethod
    def get_product_attributes(attribute, product):
        """

        :param attribute:
        :param product:
        :return:
        """
        return attribute.produkt_id == product.numer

    @staticmethod
    def get_product_id(line):
        """

        :param line:
        :return:
        """
        attribute_values = []
        for attribute in line.atrybuty.split(", "):
            attribute_values.append(
                InstallKamsERP_Configuration().xml_operand.find_attribute_values(
                    [['&', ['name', '=', attribute.split(": ")[1]],
                      ['attribute_id', '=', InstallKamsERP_Configuration().xml_operand.find_attribute(
                          [[['name', '=', attribute.split(": ")[0]]]]).get(
                          'id')]]]).get('id'))
        if len(attribute_values) > 0:
            attribute_conditions = ['&', ['name', 'ilike', line.produkt_nazwa]]
            for attribute_value in attribute_values:
                attribute_conditions.append(['attribute_value_ids', '=', attribute_value])
            product_id = InstallKamsERP_Configuration().xml_operand.find_product([attribute_conditions]).get('id')
        else:
            product_id = InstallKamsERP_Configuration().xml_operand.find_product(
                [[['name', '=', line.produkt_nazwa]]]).get('id')

        return product_id
