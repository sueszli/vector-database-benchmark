# -*- coding: utf-8 -*-
# @COPYRIGHT_begin
#
# Copyright [2015] Michał Szczygieł (m4gik), M4GiK Software
#
# @COPYRIGHT_end
import unittest
import time
from threading import Timer
import datetime

from sqlalchemy.orm import create_session

from models.kqs.kqs_order import KqsZamowienia
from models.kqs.kqs_products import KqsProdukty
from utils.database_connector import DatabaseConnector
from utils.xml_rpc_operations import _datetime_to_integer


class Test(unittest.TestCase):
    def setUp(self):
        # Create a session to use the tables
        self.session = create_session(bind=DatabaseConnector().get_engine())

    def tearDown(self):
        self.session.close()

    def test_connection(self):
        query = self.session.query(KqsZamowienia)
        result = query.all()
        print result[300].status
        self.print_some_times()

        self.assertIsNotNone(result)

    def print_time(self):
        print "From print_time", time.time()

    def print_some_times(self):
        print time.time()
        Timer(5, self.print_time, ()).start()
        Timer(10, self.print_time, ()).start()
        time.sleep(11)  # sleep while time-delay events execute
        print time.time()

    def test_fetch_products(self):
        query = self.session.query(KqsProdukty)
        timestamp_now = _datetime_to_integer(datetime.datetime.utcnow(), "%Y-%m-%d %H:%M:%S.%f")
        timestamp_yesterday = timestamp_now - 86400 * 700
        # timestamp_yesterday = datetime_to_integer(datetime.datetime.utcnow() - datetime.timedelta(700),
        #                                           "%Y-%m-%d %H:%M:%S.%f")

        result = query.filter(KqsProdukty.data < timestamp_now).filter(KqsProdukty.data > timestamp_yesterday)

        # connector = XmlRpcConnector()
        # print connector.search('product.template', [[['create_date', '<', str(datetime.datetime.now())]]])
        for record in result:
            # print connector.search('product.template', [[['record.nazwa', '<=', datetime.datetime.now()]]])
            # if not connector.search('product.template', [[['name', '=', record.nazwa]]]):
            # print datetime.datetime.fromtimestamp(record.data)
            print record.nazwa, datetime.datetime.fromtimestamp(record.data)
            self.assertIsNotNone(record)
