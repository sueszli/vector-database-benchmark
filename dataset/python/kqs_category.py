# -*- coding: utf-8 -*-
# @COPYRIGHT_begin
#
# Copyright [2015] Michał Szczygieł (m4gik), M4GiK Software
#
# @COPYRIGHT_end
from sqlalchemy import BigInteger, Column, Integer, String, text, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class KqsKategorie(Base):
    __tablename__ = u'kqs_kategorie'

    numer = Column(BigInteger, primary_key=True)
    nazwa = Column(String(100), nullable=False, server_default=text("''"))
    aktywne = Column(Integer, nullable=False, server_default=text("'0'"))
    kat_matka = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    kolejnosc = Column(BigInteger, nullable=False, server_default=text("'0'"))
    wizytowka = Column(Text, nullable=False)
    meta_tagi = Column(Text, nullable=False)
    archiwalna = Column(Integer, nullable=False, server_default=text("'0'"))
    opis_nadrzedny = Column(Text, nullable=False)
    title_strony = Column(String(200), nullable=False, server_default=text("''"))
    rozwiniecie = Column(Integer, nullable=False, server_default=text("'0'"))
    urzadzenie = Column(Integer, nullable=False, server_default=text("'0'"))
    obraz = Column(String(25), nullable=False, server_default=text("''"))
    ikona = Column(String(25), nullable=False, server_default=text("''"))
    raty = Column(Integer, nullable=False, server_default=text("'0'"))
    caraty_id = Column(String(10), nullable=False, server_default=text("''"))
