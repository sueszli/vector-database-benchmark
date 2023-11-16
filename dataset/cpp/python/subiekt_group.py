# -*- coding: utf-8 -*-
# @COPYRIGHT_begin
#
# Copyright [2015] Michał Szczygieł (m4gik), M4GiK Software
#
# @COPYRIGHT_end
from sqlalchemy import Column, Integer, String, text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class SlGrupaTw(Base):
    __tablename__ = u'sl_GrupaTw'

    grt_Id = Column(Integer, primary_key=True)
    grt_Nazwa = Column(String(50, u'Polish_CI_AS'), nullable=False, server_default=text("""\


--czas UsuwanieDefaultow end
--czas DodawanieDefaultow start

create default [DZeroString] as ''



"""))
    grt_NrAnalityka = Column(String(3, u'Polish_CI_AS'), server_default=text("""\


--czas UsuwanieDefaultow end
--czas DodawanieDefaultow start

create default [DZeroString] as ''



"""))
