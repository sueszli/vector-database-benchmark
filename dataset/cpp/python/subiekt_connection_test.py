# -*- coding: utf-8 -*-
# @COPYRIGHT_begin
#
# Copyright [2015] Michał Szczygieł (m4gik), M4GiK Software
#
# @COPYRIGHT_end
import unittest
from socket import *

from sqlalchemy import MetaData, desc
from sqlalchemy.orm import create_session

from configs.kamserp_config import SUBIEKT_DATABASE_CONNECTOR, SUBIEKT_DATABASE_USER, \
    SUBIEKT_DATABASE_PASSWORD, \
    SUBIEKT_DATABASE_ADDRESS, SUBIEKT_DATABASE_DATABASE_NAME, SUBIEKT_DATABASE_DRIVER, SUBIEKT_DATABASE_PORT

from models.subiekt.subiekt_product import TwStan, TwTowar, TwCena, DokPozycja, DokDokument

from utils.database_connector import DatabaseConnector
from utils.xml_rpc_operations import convert_decimal_to_float


class Test(unittest.TestCase):
    def setUp(self):
        self.dbobject = DatabaseConnector(connector=SUBIEKT_DATABASE_CONNECTOR, user=SUBIEKT_DATABASE_USER,
                                          password=SUBIEKT_DATABASE_PASSWORD,
                                          host=SUBIEKT_DATABASE_ADDRESS + ':' + SUBIEKT_DATABASE_PORT,
                                          dbname=SUBIEKT_DATABASE_DATABASE_NAME, driver=SUBIEKT_DATABASE_DRIVER,
                                          charset=None)
        # Create a session to use the tables
        self.session = create_session(bind=self.dbobject.get_engine())

    def tearDown(self):
        self.session.close()

    def test_connection(self):
        meta = MetaData()
        meta.reflect(bind=self.dbobject.get_engine())

        for table in meta.tables.values():
            print """
        class %s(Base):
            __table__ = Table(%r, Base.metadata, autoload=True)

        """ % (table.name, table.name)

    def test_product(self):
        query = self.session.query(TwTowar).filter(TwTowar.tw_PodstKodKresk == '3295249124601')
        result = query.first()

        query = self.session.query(TwCena).filter(TwCena.tc_IdTowar == result.tw_Id)
        price = query.first()

        query = self.session.query(TwStan).filter(TwStan.st_TowId == result.tw_Id)
        stan = query.first()
        print result.tw_Nazwa, price.tc_CenaNetto1, convert_decimal_to_float(stan.st_Stan)

        self.assertIsNotNone(result)

    def test_wartosc_netto(self):
        query = self.session.query(TwTowar).filter(TwTowar.tw_PodstKodKresk == '3295249124601')
        towar = query.first()

        document = self.session.query(DokDokument, DokPozycja) \
            .outerjoin(DokPozycja, DokDokument.dok_Id == DokPozycja.ob_DokHanId).filter(DokDokument.dok_Typ == 1) \
            .order_by(desc(DokDokument.dok_DataWyst)).filter(DokPozycja.ob_TowId == towar.tw_Id).first()
        print document[1].ob_CenaNetto

        query = self.session.query(TwTowar).filter(TwTowar.tw_PodstKodKresk == '5907522912581')
        towar = query.first()

        document = self.session.query(DokDokument, DokPozycja) \
            .outerjoin(DokPozycja, DokDokument.dok_Id == DokPozycja.ob_DokHanId).filter(DokDokument.dok_Typ == 1) \
            .order_by(desc(DokDokument.dok_DataWyst)).filter(DokPozycja.ob_TowId == towar.tw_Id).first()
        print document[1].ob_CenaNetto

    def test_check_port(self):
        """
        Checks available ports.
        """
        fTimeOutSec = 5.0
        sNetworkAddress = '10.0.0.54'
        aiHostAddresses = range(1, 255)
        aiPorts = range(1, 65535)

        setdefaulttimeout(fTimeOutSec)
        print "Starting Scan..."
        # for h in aiHostAddresses:
        for p in aiPorts:
            s = socket(AF_INET, SOCK_STREAM)
            # address = ('%s.%d' % (sNetworkAddress, h))
            address = ('%s' % sNetworkAddress)
            result = s.connect_ex((address, p))
            if 0 == result:
                print "%s:%d - OPEN (%d)" % (address, p, result)
            elif 10035 == result:
                # do nothing, was a timeout, probably host doesn't exist
                pass
            else:
                # print "%s:%d - closed (%d)" % (address, p, result)
                pass
                s.close()
        print "Scan Completed."
