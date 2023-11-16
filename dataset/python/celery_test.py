# -*- coding: utf-8 -*-
# @COPYRIGHT_begin
#
# Copyright [2015] Michał Szczygieł (m4gik), M4GiK Software
#
# @COPYRIGHT_end
import json
import unittest

from utils.xml_rpc_connector import XmlRpcConnector


class Test(unittest.TestCase):
    def setUp(self):
        # Create a session to use the tables
        self.connector = XmlRpcConnector()
        self.id_value = 0

    def tearDown(self):
        pass

    def test_CRUD_operation(self):
        partner_record = [{
            'name': 'Fabien2',
            'email': 'example@odoo.com'
        }]

        result = self.connector.create('res.partner', partner_record)
        self.assertNotEqual(self.id, result, "Is different then 0")
        id_record = result

        print json.dumps(self.connector.read('res.partner', [[id_record]]), indent=4, sort_keys=True)

        self.connector.delete('res.partner', [[id_record]])
        self.assertEqual([], self.connector.search('res.partner', [[['id', '=', id_record]]]))
