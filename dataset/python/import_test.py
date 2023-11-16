# -*- coding: utf-8 -*-
# @COPYRIGHT_begin
#
# Copyright [2016] Michał Szczygieł (m4gik), M4GiK Software
#
# @COPYRIGHT_end
import base64
import json
import unittest
import urllib2
from sqlalchemy import asc
from sqlalchemy.orm import create_session
from configs.kamserp_config import SUBIEKT_DATABASE_CONNECTOR, SUBIEKT_DATABASE_USER, SUBIEKT_DATABASE_PORT, \
    SUBIEKT_DATABASE_DRIVER, SUBIEKT_DATABASE_DATABASE_NAME, SUBIEKT_DATABASE_ADDRESS, SUBIEKT_DATABASE_PASSWORD
from models.kqs.kqs_category import KqsKategorie
from models.kqs.kqs_images import KqsGaleriaZaczepy, KqsGaleria
from models.kqs.kqs_manufacturer import KqsProducenci
from models.kqs.kqs_products import KqsProdukty
from models.kqs.kqs_products_attribute import KqsProduktyAtrybuty, KqsProduktyOpcje, KqsProduktyWartosci
from models.kqs.kqs_products_category import KqsProduktyKategorie
from models.subiekt.subiekt_product import TwTowar, TwStan, TwCena
from utils.database_connector import DatabaseConnector

from utils.xml_rpc_connector import XmlRpcConnector
from utils.xml_rpc_operations import XmlRpcOperations, convert_decimal_to_float


class Test(unittest.TestCase):
    def setUp(self):
        # Create a session to use the XML RPC Connector
        self.connector = XmlRpcConnector()
        self.xml_operand = XmlRpcOperations()
        self.id_value = 0

        # Create a session to use the tables
        # self.session = create_session(bind=DatabaseConnector().get_engine())
        self.session = create_session(bind=DatabaseConnector(dbname="kamsbhp_sklep2").get_engine())

    def tearDown(self):
        self.session.close()

    def test_KQS_operation_to_Odoo(self):
        query = self.session.query(KqsProdukty).filter(KqsProdukty.kod_kreskowy == '3295249124601')
        kqs_product = query.first()

        query = self.session.query(KqsGaleriaZaczepy).filter(
            KqsGaleriaZaczepy.produkt_id == kqs_product.numer).filter(
            KqsGaleriaZaczepy.kolejnosc == 1)
        image = query.first()
        query = self.session.query(KqsGaleria).filter(
            KqsGaleria.numer == image.obraz_id)

        image_name = query.first()
        image = "http://kams.com.pl//galerie/" + image_name.obraz[0] + "/" + image_name.obraz + ".jpg"
        image_small = "http://kams.com.pl//galerie/" + image_name.obraz[0] + "/" + image_name.obraz + "_k.jpg"
        image_medium = "http://kams.com.pl//galerie/" + image_name.obraz[0] + "/" + image_name.obraz + "_m.jpg"

        dbobject = DatabaseConnector(connector=SUBIEKT_DATABASE_CONNECTOR, user=SUBIEKT_DATABASE_USER,
                                     password=SUBIEKT_DATABASE_PASSWORD,
                                     host=SUBIEKT_DATABASE_ADDRESS + ':' + SUBIEKT_DATABASE_PORT,
                                     dbname=SUBIEKT_DATABASE_DATABASE_NAME, driver=SUBIEKT_DATABASE_DRIVER,
                                     charset=None)

        # Create a session for subiekt
        subiekt_session = create_session(bind=dbobject.get_engine())
        query = subiekt_session.query(TwTowar).filter(TwTowar.tw_PodstKodKresk == '3295249124601')
        result = query.first()
        query = subiekt_session.query(TwCena).filter(TwCena.tc_IdTowar == result.tw_Id)
        price = query.first()
        query = subiekt_session.query(TwStan).filter(TwStan.st_TowId == result.tw_Id)
        stan = query.first()
        subiekt_session.close()

        category_to_insert = [{
            'name': 'Test',
            'kqs_original_id': 999
        }]
        category = self.connector.create('category', category_to_insert)

        query = self.session.query(KqsProducenci).filter(KqsProducenci.nazwa == 'JHK')
        kqs_manufacturer = query.first()

        image_manufacturer = "http://kams.com.pl/galerie/producenci/" + kqs_manufacturer.logo_producenta

        manufacturer_to_insert = [{
            'name': kqs_manufacturer.nazwa,
            'image': base64.encodestring(urllib2.urlopen(image_manufacturer).read()),
        }]

        manufacturer = self.connector.create('kams_erp.manufacturer', manufacturer_to_insert)

        product_template = [{
            'name': kqs_product.nazwa,
            'description': kqs_product.opis,
            'price': float(kqs_product.cena),
            'price_subiekt': float(price.tc_CenaBrutto1),
            'barcode': str(kqs_product.kod_kreskowy),
            'description_sale': kqs_product.krotki_opis,
            'image': base64.encodestring(urllib2.urlopen(image).read()),
            'image_medium': base64.encodestring(urllib2.urlopen(image_medium).read()),
            'image_small': base64.encodestring(urllib2.urlopen(image_small).read()),
            'warehouse_id': str(1),
            'manufacturer_id': manufacturer,
            'amount': convert_decimal_to_float(stan.st_Stan),
            'weight': convert_decimal_to_float(kqs_product.waga),
            'categ_id': self.connector.read('kams_erp.category', [[category]])[0].get('categ_id')[0],
        }]

        products = self.connector.create('kams_erp.product', product_template)
        self.assertNotEqual(self.id, products, "Is different then 0")
        id_record = products

        print json.dumps(self.connector.read('kams_erp.product', [[id_record]]), indent=4, sort_keys=True)

        self.connector.delete('kams_erp.product', [[id_record]])
        self.connector.delete('kams_erp.category', [[category]])
        self.connector.delete('kams_erp.manufacturer', [[manufacturer]])
        self.assertEqual([], self.connector.search('kams_erp.product', [[['id', '=', id_record]]]))

    def test_import_category(self):
        query = self.session.query(KqsKategorie).order_by(asc(KqsKategorie.numer))
        categories = query.all()
        kqs_category = categories[14]

        if int(kqs_category.kat_matka) != 0:
            parent_kqs = self.session.query(KqsKategorie).filter(
                KqsKategorie.numer == kqs_category.kat_matka).first()
            parent_odoo = self.xml_operand.find_category([[['name', '=', parent_kqs.nazwa]]])
            if not parent_odoo:
                category_to_insert = [{
                    'name': parent_kqs.nazwa,
                }]
                parent_odoo = self.xml_operand.insert_category(category_to_insert)

            category_to_insert = [{
                'name': kqs_category.nazwa,
                'parent_id': parent_odoo.get('categ_id')[0],
            }]
            category = self.xml_operand.insert_category(category_to_insert)

            print json.dumps(self.connector.read('kams_erp.category', [[category.get('id')]]), indent=4, sort_keys=True)

    def test_insert_category(self):
        query = self.session.query(KqsKategorie).filter(KqsKategorie.nazwa == 'Produkty archiwalne')
        kqs_category = query.first()

        category_to_insert = [{
            'name': kqs_category.nazwa,
        }]

        inserted = self.xml_operand.insert_category(category_to_insert)
        self.connector.delete('kams_erp.category', [[inserted.get('id')]])
        self.assertEqual([], self.connector.search('kams_erp.category', [[['id', '=', inserted.get('id')]]]))

    def test_warehouse(self):
        print json.dumps(self.connector.read('stock.warehouse', [[1]]), indent=4,
                         sort_keys=True)

    def test_manufacturer(self):
        query = self.session.query(KqsProducenci).filter(KqsProducenci.nazwa == 'JHK')
        kqs_manufacturer = query.first()

        image = "http://kams.com.pl/galerie/producenci/" + kqs_manufacturer.logo_producenta

        manufacturer_to_insert = [{
            'name': kqs_manufacturer.nazwa,
            'image': base64.encodestring(urllib2.urlopen(image).read()),
        }]

        manufacturer = self.connector.create('kams_erp.manufacturer', manufacturer_to_insert)
        print json.dumps(self.connector.read('kams_erp.manufacturer', [[manufacturer]]), indent=4, sort_keys=True)

    def test_products_category(self):
        query = self.session.query(KqsProduktyKategorie)
        kqs_category = query.first()

        print kqs_category.produkt_id

    def test_find_category(self):
        query = self.session.query(KqsKategorie).filter(KqsKategorie.nazwa == 'Produkty archiwalne')
        kqs_category = query.first()

        category_to_insert = [{
            'name': kqs_category.nazwa,
            'kqs_original_id': kqs_category.numer,
        }]

        inserted = self.xml_operand.insert_category(category_to_insert)
        odoo_category = self.xml_operand.find_category([[['kqs_original_id', '=', kqs_category.numer]]])

        print odoo_category

        self.connector.delete('product.category', [[inserted.get('id')]])
        self.assertEqual([], self.connector.search('kams_erp.category', [[['id', '=', inserted.get('id')]]]))

    def test_extract_category(self):
        query = self.session.query(KqsProducenci)
        manufacturers = query.all()

        manufacturer = next((manufacturer for manufacturer in manufacturers if manufacturer.nazwa == 'JHK'))
        print manufacturer

    def test_attribute_insertion(self):
        kqs_product = self.session.query(KqsProdukty).filter(KqsProdukty.kod_kreskowy == '3295249035921').first()
        attributes = self.session.query(KqsProduktyAtrybuty).filter(
            KqsProduktyAtrybuty.produkt_id == kqs_product.numer).all()
        product = self.xml_operand.find_product([[['barcode', '=', str(3295249035921)]]])
        for attribute in attributes:
            options = self.session.query(KqsProduktyOpcje).filter(KqsProduktyOpcje.numer == attribute.opcja_id).all()
            for option in options:
                odoo_attribute = self.xml_operand.find_attribute([[['kqs_original_id', '=', option.numer]]])
                if not odoo_attribute:
                    attribute_to_insert = [{
                        'name': option.opcja,
                        'sequence': option.kolejnosc,
                        'kqs_original_id': option.numer,
                    }]
                    odoo_attribute = self.xml_operand.insert_attribute(attribute_to_insert)
                values = self.session.query(KqsProduktyWartosci).filter(
                    KqsProduktyWartosci.numer == attribute.wartosc_id).all()
                for value in values:
                    odoo_attribute_value = self.xml_operand.find_attribute_values(
                        [[['kqs_original_id', '=', value.numer]]])
                    if not odoo_attribute_value:
                        attribute_value_to_insert = [{
                            'name': value.wartosc,
                            'price_extra': float(value.zmiana_wartosc),
                            'attribute_id': odoo_attribute.get('attribute_id')[0],
                            'kqs_original_id': value.numer,
                            # 'product_ids': [(6, 0, [product.get('product_id')[0]])],
                        }]
                        odoo_attribute_value = self.xml_operand.insert_attribute_values(attribute_value_to_insert)

                    odoo_attribute_line = self.xml_operand.find_attribute_line(
                        [['&', ['product_tmpl_id', '=', product.get('product_id')[0]],
                          ['value_ids', 'ilike', odoo_attribute_value.get('attribute_value_id')[0]]]])
                    print odoo_attribute_line
                    if not odoo_attribute_line:
                        attribute_line_to_insert = [{
                            'product_tmpl_id': product.get('product_id')[0],
                            'attribute_id': odoo_attribute.get('attribute_id')[0],
                            'value_ids': [(6, 0, [odoo_attribute_value.get('attribute_value_id')[0]])]
                        }]
                        attribute_line = self.xml_operand.insert_attribute_line(attribute_line_to_insert)
                        print attribute_line
