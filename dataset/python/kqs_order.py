# -*- coding: utf-8 -*-
# @COPYRIGHT_begin
#
# Copyright [2016] Michał Szczygieł (m4gik), M4GiK Software
#
# @COPYRIGHT_end


__author__ = 'm4gik'

from sqlalchemy import BigInteger, Column, Float, Integer, String, Text, text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class KqsZamowienia(Base):
    __tablename__ = u'kqs_zamowienia'

    id = Column(BigInteger, primary_key=True)
    cena = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    przesylka = Column(BigInteger, nullable=False, server_default=text("'0'"))
    u_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))
    status = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    powod = Column(Text, nullable=False)
    waga = Column(Float(8, True), nullable=False, server_default=text("'0.00'"))
    klient_email = Column(String(100), nullable=False, server_default=text("''"))
    klient_telefon = Column(String(20), nullable=False, server_default=text("''"))
    waluta_jednostka = Column(String(15), nullable=False, server_default=text("''"))
    waluta_przelicznik = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    uwagi = Column(Text, nullable=False)
    user_kup_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    admin_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    last_admin_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    last_admin_data = Column(BigInteger, nullable=False, server_default=text("'0'"))
    status_wplata = Column(String(11), nullable=False, server_default=text("''"))
    status_produkcja = Column(String(11), nullable=False, server_default=text("''"))
    status_wyprodukowano = Column(String(11), nullable=False, server_default=text("''"))
    numer_listu = Column(String(50), nullable=False, server_default=text("''"))
    data_wyslania = Column(BigInteger, nullable=False, server_default=text("'0'"))
    dokument = Column(Integer, nullable=False, server_default=text("'0'"))
    przesylka_nazwa = Column(String(100), nullable=False, server_default=text("''"))
    przesylka_koszt = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    przesylka_koszt_brutto = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    przesylka_opis = Column(Text, nullable=False)
    przesylka_fra = Column(Integer, nullable=False, server_default=text("'0'"))
    kwota_platnosci = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    przelew = Column(Integer, nullable=False, server_default=text("'0'"))
    gotowka = Column(Integer, nullable=False, server_default=text("'0'"))
    potwierdzenie = Column(Integer, nullable=False, server_default=text("'0'"))
    klient_osoba = Column(String(100), nullable=False)
    klient_firma = Column(String(255), nullable=False)
    klient_nip = Column(String(50), nullable=False)
    klient_wojewodztwo = Column(String(100), nullable=False)
    klient_kraj = Column(String(100), nullable=False)
    rabat_model = Column(Integer, nullable=False, server_default=text("'0'"))
    status_dodatkowy = Column(BigInteger, nullable=False, server_default=text("'0'"))
    przesylka_vat = Column(Integer, nullable=False, server_default=text("'0'"))
    klient_ulica = Column(String(100), nullable=False)
    klient_dom = Column(String(20), nullable=False, server_default=text("''"))
    klient_kod = Column(String(10), nullable=False, server_default=text("''"))
    klient_miasto = Column(String(100), nullable=False, server_default=text("''"))
    wysylka_odbiorca = Column(String(200), nullable=False, server_default=text("''"))
    wysylka_ulica = Column(String(100), nullable=False, server_default=text("''"))
    wysylka_dom = Column(String(20), nullable=False, server_default=text("''"))
    wysylka_kod = Column(String(10), nullable=False, server_default=text("''"))
    wysylka_miasto = Column(String(100), nullable=False, server_default=text("''"))
    referent = Column(String(255), nullable=False, server_default=text("''"))
    bon_kod = Column(String(35), nullable=False, server_default=text("''"))
    bon_wartosc = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    status_przypomnienie = Column(String(11), nullable=False, server_default=text("''"))
    unikalny_numer = Column(String(20), nullable=False, server_default=text("''"))
    klient_ip = Column(String(50), nullable=False, server_default=text("''"))
    waluta_kod = Column(String(5), nullable=False, server_default=text("''"))
    odroczenie = Column(Integer, nullable=False, server_default=text("'0'"))
    leasing = Column(Integer, nullable=False, server_default=text("'0'"))
    jezyk = Column(String(32), nullable=False, server_default=text("''"))
    rodzaj_cen = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsZamowieniaProdukty(Base):
    __tablename__ = u'kqs_zamowienia_produkty'

    numer = Column(BigInteger, primary_key=True)
    zamowienie_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    produkt_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    produkt_nazwa = Column(String(200), nullable=False, server_default=text("''"))
    ilosc = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    jm = Column(String(50), nullable=False, server_default=text("''"))
    cena = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    podatek = Column(Integer, nullable=False, server_default=text("'0'"))
    atrybuty = Column(Text, nullable=False)
    atrybuty_magazyn = Column(Text, nullable=False)
    pkwiu = Column(String(50), nullable=False, server_default=text("''"))
    kod_produktu = Column(String(50), nullable=False, server_default=text("''"))
    kod_dostawcy = Column(String(50), nullable=False, server_default=text("''"))
    kod_kreskowy = Column(String(50), nullable=False, server_default=text("''"))
    kod_plu = Column(String(50), nullable=False, server_default=text("''"))
    rabat = Column(Float(4, True), nullable=False, server_default=text("'0.00'"))
    kod_rabatowy = Column(String(50), nullable=False, server_default=text("''"))
    kod_producenta = Column(String(50), nullable=False, server_default=text("''"))
