# -*- coding: utf-8 -*-
# @COPYRIGHT_begin
#
# Copyright [2015] Michał Szczygieł (m4gik), M4GiK Software
#
# @COPYRIGHT_end

from sqlalchemy import BigInteger, Column, Integer, String, text, SmallInteger, Table
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class KqsGaleria(Base):
    __tablename__ = u'kqs_galeria'

    numer = Column(BigInteger, primary_key=True)
    obraz = Column(String(100), nullable=False, server_default=text("''"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))
    wysokosc = Column(SmallInteger, nullable=False, server_default=text("'0'"))
    szerokosc = Column(SmallInteger, nullable=False, server_default=text("'0'"))
    rozszerzenie = Column(String(5), nullable=False, server_default=text("''"))


t_kqs_galeria_import = Table(
    u'kqs_galeria_import', metadata,
    Column(u'plik', String(100), nullable=False, server_default=text("''")),
    Column(u'zaczep', BigInteger, nullable=False, server_default=text("'0'"))
)


class KqsGaleriaZaczepy(Base):
    __tablename__ = u'kqs_galeria_zaczepy'

    numer = Column(BigInteger, primary_key=True)
    obraz_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    produkt_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    kolejnosc = Column(Integer, nullable=False, server_default=text("'0'"))
