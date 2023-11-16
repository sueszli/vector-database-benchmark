from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import Column, String, INTEGER, VARCHAR, DATE, DateTime, ForeignKey, FLOAT
Base = declarative_base()

class FundBaseInfoModel(Base):
    __tablename__ = 'LOF_BaseInfo'
    id = Column(INTEGER, primary_key=True, autoincrement=True)
    code = Column(VARCHAR(6), comment='基金代码', unique=True)
    name = Column(VARCHAR(40), comment='基金名称')
    category = Column(VARCHAR(8), comment='基金类别')
    invest_type = Column(VARCHAR(6), comment='投资类别')
    manager_name = Column(VARCHAR(48), comment='管理人呢名称')
    issue_date = Column(DATE, comment='上市日期')
    child = relationship('ShareModel')

    def __str__(self):
        if False:
            while True:
                i = 10
        return f'<{self.code}><{self.name}>'

class ShareModel(Base):
    __tablename__ = 'LOF_Share'
    id = Column(INTEGER, primary_key=True, autoincrement=True)
    code = Column(VARCHAR(6), ForeignKey('LOF_BaseInfo.code'), comment='代码')
    date = Column(DATE, comment='份额日期')
    share = Column(FLOAT, comment='份额 单位：万份')
    parent = relationship('FundBaseInfoModel')
    crawltime = Column(DateTime, comment='爬取日期')