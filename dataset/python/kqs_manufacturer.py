# -*- coding: utf-8 -*-
# @COPYRIGHT_begin
#
# Copyright [2016] Michał Szczygieł (m4gik), M4GiK Software
#
# @COPYRIGHT_end

from sqlalchemy import BigInteger, Column, String, Text, text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class KqsProducenci(Base):
    __tablename__ = u'kqs_producenci'

    numer = Column(BigInteger, primary_key=True)
    nazwa = Column(String(100), nullable=False, unique=True, server_default=text("''"))
    wizytowka = Column(Text, nullable=False)
    logo_producenta = Column(String(100), nullable=False, server_default=text("''"))
    wizytowka_2 = Column(Text, nullable=False)