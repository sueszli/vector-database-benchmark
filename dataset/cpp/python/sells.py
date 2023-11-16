from db.base_class import Base
from sqlalchemy import Column, String, DateTime,Integer, Float, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

class Sells(Base):
    id = Column(Integer, primary_key=True ,nullable=False, autoincrement=True)
    id_sell = Column(Integer,unique=False ,nullable=False)
    id_product = Column(String, ForeignKey('product.key'), nullable=False)
    amount = Column(Float, nullable=False)
    sell_price = Column(Float, nullable=False)
    buy_price = Column(Float, nullable=False)
    total = Column(Float, nullable=False)
    date = Column(DateTime, default=datetime.now(), nullable=False)
    producto = relationship("Product", backref="Sells")