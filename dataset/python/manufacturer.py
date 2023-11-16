# -*- coding: utf-8 -*-
# @COPYRIGHT_begin
#
# Copyright [2015] Michał Szczygieł (m4gik), M4GiK Software
#
# @COPYRIGHT_end
import base64
import urllib2

from kams_erp.configs.install.configuration import InstallKamsERP_Configuration


class InstallKamsERP_Manufacturer(InstallKamsERP_Configuration):

    def insert_or_find_manufacturer(self, manufacturers, producent_id, is_supplier):
        """

        :param manufacturers:
        :param producent_id:
        :param is_supplier:
        :return:
        """
        odoo_manufacturer = self.xml_operand.find_partner([[['kqs_original_id', '=', producent_id]]])
        if not odoo_manufacturer:
            manufacturer = next((manufacturer for manufacturer in manufacturers if manufacturer.numer == producent_id))
            try:
                image_manufacturer = "http://kams.com.pl/galerie/producenci/" + manufacturer.logo_producenta
                manufacturer_to_insert = [{
                    'name': manufacturer.nazwa,
                    'kqs_original_id': producent_id,
                    'image': base64.encodestring(urllib2.urlopen(image_manufacturer).read()),
                    'supplier': is_supplier
                }]
            except (urllib2.HTTPError, AttributeError):
                manufacturer_to_insert = [{
                    'name': manufacturer.nazwa,
                    'kqs_original_id': producent_id,
                    'image': base64.encodestring(
                        urllib2.urlopen(self.url + '/base/static/src/img/company_image.png').read()),
                    'supplier': is_supplier
                }]
            manufacturer_id = self.xml_operand.insert_partner(manufacturer_to_insert).get('id')
        else:
            manufacturer_id = odoo_manufacturer.get('id')
        return manufacturer_id
