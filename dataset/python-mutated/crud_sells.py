from datetime import datetime
import psycopg2
from fastapi.encoders import jsonable_encoder
from typing import Optional
from core.config import settings
from schemas.poducts import Product as ProductSchema
from schemas.sells import Sell as SellSchema
from schemas.sells import SellProduct as SellProductSchema

class CRUDSells:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.connected = False
        self.conn = None
        self.cursor = None
        self.headers_sell = ['id', 'id_sell', 'id_product', 'amount', 'sell_price', 'buy_price', 'total', 'date']
        self.headers_product = ['key', 'code', 'codebar', 'codebarInner', 'codebarMaster', 'unit', 'description', 'brand', 'buy', 'retailsale', 'wholesale', 'inventory', 'min_inventory', 'department', 'id', 'box', 'master', 'lastUpdate']

    def OpenConnection(self):
        if False:
            print('Hello World!')
        self.conn = psycopg2.connect(database=settings.POSTGRES_DB, host=settings.POSTGRES_SERVER, user=settings.POSTGRES_USER, password=settings.POSTGRES_PASSWORD, port=settings.POSTGRES_PORT)
        self.cursor = self.conn.cursor()
        self.connected = True

    def create_sell(self, products: list[SellProductSchema]):
        if False:
            return 10
        try:
            prices = {}
            for p in products:
                self.cursor.execute(f"SELECT * FROM product WHERE key='{p.key}'")
                obj_out = self.cursor.fetchone()
                if obj_out:
                    obj_out = {x: y for (x, y) in zip(self.headers_product, obj_out)}
                    obj_out = ProductSchema(**obj_out)
                prices[p.key] = {'amount': p.amount, 'sell_price': obj_out.retailsale if p.retail else obj_out.wholesale, 'buy_price': obj_out.buy}
            consulta = 'INSERT INTO sells (id_sell,id_product, amount, sell_price,buy_price,total,date) VALUES '
            id_generated = int(datetime.now().timestamp())
            valores = ', '.join([f"('{id_generated}','{p}', '{prices[p]['amount']}', '{prices[p]['sell_price']}', '{prices[p]['buy_price']}','{prices[p]['amount'] * prices[p]['sell_price']}','{datetime.now()}')" for p in prices.keys()])
            consulta += valores + ';'
            self.cursor.execute(consulta)
            self.conn.commit()
            return {'mensaje': 'Sell send succesfully', 'status_code': 200}
        except:
            return {'mensaje': 'Error', 'status_code': 404}

    def get_sell(self, id_sell: int) -> list[SellSchema]:
        if False:
            for i in range(10):
                print('nop')
        self.cursor.execute(f"SELECT * FROM sells WHERE id_sell='{id_sell}'")
        sells = []
        if self.cursor and self.cursor.rowcount > 0:
            obj_out = self.cursor.fetchall()
            if obj_out:
                for product in obj_out:
                    p = {x: y for (x, y) in zip(self.headers_sell, product)}
                    p = SellSchema(**p)
                    sells.append(p)
        return sells

    def CloseConnection(self):
        if False:
            while True:
                i = 10
        self.conn.rollback()
        self.cursor.close()
        self.conn.close()
        self.conn = None
        self.cursor = None
        self.connected = False
CRUDsellsObject = CRUDSells()