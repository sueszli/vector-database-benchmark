# -*- coding: utf-8 -*-
# @COPYRIGHT_begin
#
# Copyright [2016] Michał Szczygieł (m4gik), M4GiK Software
#
# @COPYRIGHT_end

from sqlalchemy import asc
from kams_erp.configs.install.configuration import InstallKamsERP_Configuration
from kams_erp.configs.install.product import InstallKamsERP_Product

from kams_erp.models.kqs_manufacturer import KqsProducenci
from kams_erp.models.kqs_products import KqsProdukty
from kams_erp.models.kqs_products_attribute import KqsProduktyAtrybuty
from kams_erp.models.kqs_products_category import KqsProduktyKategorie
from kams_erp.models.kqs_category import KqsKategorie
from kams_erp.models.kqs_supplier import KqsDostawcy
from kams_erp.models.subiekt_product import TwTowar, TwStan, TwCena


class InstallKamsERP(InstallKamsERP_Configuration):
    """
    Class responsible for install all required data.
    """

    def install_data_from_kqs(self, insert_all=False):
        """
        Install data from KQS databse to Odoo database integrated with Subiekt.
        :param insert_all: When is False, insert only product have relation in subiekt database.
        """
        InstallKamsERP_Product().insert_transport_as_product()
        self.__insert_all_product_with_complete_data(insert_all)

    def __insert_all_product_with_complete_data(self, insert_all=False):
        # This fetching all data could be heavy, but operations have low affect on database.
        products = self.session.query(KqsProdukty).all()  # filter(KqsProdukty.numer == 64)
        categories = self.session.query(KqsKategorie).order_by(asc(KqsKategorie.numer)).all()
        kqs_products_category = self.session.query(KqsProduktyKategorie).all()
        manufacturers = self.session.query(KqsProducenci).all()
        suppliers = self.session.query(KqsDostawcy).all()
        attributes = self.session.query(KqsProduktyAtrybuty).all()

        for product in products:
            category_product = next((category_product for category_product in kqs_products_category if
                                     category_product.produkt_id == product.numer))
            if category_product is not None:
                try:
                    category = next(
                        (category for category in categories if category_product.kategoria_id == category.numer))
                except StopIteration:
                    category = None
                if insert_all:
                    if category is not None:
                        if not self.xml_operand.find_product([[['unique_product_number', '=', product.kod_produktu]]]):
                            if product.kod_kreskowy != '':
                                # Create a session for subiekt.
                                query = self.subiekt_session.query(TwTowar).filter(
                                    TwTowar.tw_PodstKodKresk == product.kod_kreskowy)
                                result = query.first()
                                if result is not None:
                                    query = self.subiekt_session.query(TwCena).filter(TwCena.tc_IdTowar == result.tw_Id)
                                    price = query.first()
                                    query = self.subiekt_session.query(TwStan).filter(TwStan.st_TowId == result.tw_Id)
                                    stan = query.first()
                                    self.subiekt_session.close()
                                    # End a session for subiekt.
                                    InstallKamsERP_Product().insert_product(product, category, manufacturers, suppliers,
                                                                            attributes, price.tc_CenaBrutto1,
                                                                            stan.st_Stan)
                            else:
                                if product:
                                    InstallKamsERP_Product().insert_product(product, category, manufacturers, suppliers,
                                                                            attributes)
                else:
                    if category is not None and product.kod_kreskowy != '':
                        if not self.xml_operand.find_product([[['barcode', '=', product.kod_kreskowy]]]):
                            # Create a session for subiekt.
                            query = self.subiekt_session.query(TwTowar).filter(
                                TwTowar.tw_PodstKodKresk == product.kod_kreskowy)
                            result = query.first()
                            if result is not None:
                                query = self.subiekt_session.query(TwCena).filter(TwCena.tc_IdTowar == result.tw_Id)
                                price = query.first()
                                query = self.subiekt_session.query(TwStan).filter(TwStan.st_TowId == result.tw_Id)
                                stan = query.first()
                                self.subiekt_session.close()
                            else:
                                continue
                            # End a session for subiekt.

                            InstallKamsERP_Product().insert_product(product, category, manufacturers, suppliers,
                                                                    attributes, price.tc_CenaBrutto1, stan.st_Stan)


InstallKamsERP().install_data_from_kqs(True)
# InstallKamsERP().get_orders()
