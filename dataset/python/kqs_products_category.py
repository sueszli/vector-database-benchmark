# -*- coding: utf-8 -*-
# @COPYRIGHT_begin
#
# Copyright [2016] Michał Szczygieł (m4gik), M4GiK Software
#
# @COPYRIGHT_end
from sqlalchemy import BigInteger, Column, text, PrimaryKeyConstraint
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class KqsProduktyKategorie(Base):
    __tablename__ = u'kqs_produkty_kategorie'
    __table_args__ = (
        PrimaryKeyConstraint(u'produkt_id', u'kategoria_id'),
    )

    produkt_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    kategoria_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
