# -*- coding: utf-8 -*-
# @COPYRIGHT_begin
#
# Copyright [2016] Michał Szczygieł (m4gik), M4GiK Software
#
# @COPYRIGHT_end
from sqlalchemy import BigInteger, Column, Float, Index, Integer, SmallInteger, String, text, Table
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class KqsProduktyAtrybuty(Base):
    __tablename__ = u'kqs_produkty_atrybuty'
    __table_args__ = (
        Index(u'opcja_wartosc', u'opcja_id', u'wartosc_id'),
    )

    numer = Column(BigInteger, primary_key=True)
    produkt_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    opcja_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    wartosc_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    zmiana_ceny = Column(Integer, nullable=False, server_default=text("'0'"))
    zmiana_wartosc = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    zmiana_jednostka = Column(Integer, nullable=False, server_default=text("'0'"))
    zmiana_info = Column(Integer, nullable=False, server_default=text("'0'"))
    widocznosc = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsProduktyWartosci(Base):
    __tablename__ = u'kqs_produkty_wartosci'

    numer = Column(BigInteger, primary_key=True)
    wartosc = Column(String(255), nullable=False, index=True, server_default=text("''"))
    kolejnosc = Column(SmallInteger, nullable=False, server_default=text("'0'"))
    zmiana_ceny = Column(Integer, nullable=False, server_default=text("'0'"))
    zmiana_wartosc = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    zmiana_jednostka = Column(Integer, nullable=False, server_default=text("'0'"))
    produkt_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    opis = Column(String(255), nullable=False, server_default=text("''"))


t_kqs_produkty_wersje = Table(
    u'kqs_produkty_wersje', metadata,
    Column(u'produkt_id', BigInteger, nullable=False, index=True, server_default=text("'0'")),
    Column(u'wersja_id', BigInteger, nullable=False, index=True, server_default=text("'0'"))
)


class KqsProduktyOpcje(Base):
    __tablename__ = u'kqs_produkty_opcje'

    numer = Column(BigInteger, primary_key=True)
    opcja = Column(String(255), nullable=False, server_default=text("''"))
    opis = Column(String(255), nullable=False, server_default=text("''"))
    kolejnosc = Column(SmallInteger, nullable=False, server_default=text("'0'"))
    magazyn = Column(Integer, nullable=False, server_default=text("'0'"))
    format = Column(Integer, nullable=False, server_default=text("'0'"))

