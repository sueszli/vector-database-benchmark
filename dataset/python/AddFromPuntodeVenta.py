import math
import time
from datetime import datetime
import pandas as pd
import numpy
import random

from crud.crud_products import CRUDproductsObject
from schemas.poducts import Product as ProductSchema


def addData():
    db = CRUDproductsObject

    # db.OpenConnection()

    # cursor.execute("DELETE FROM product *")
    # conn.commit()

    file_path = "C:/Users/gilis/Desktop/bdcsv.csv"
    data = pd.read_csv(file_path)
    start_time = time.time()

    for index, row in data.iterrows():
        # print(row['Codigo'] + " " + row['Descripcion'])
        try:
            db.OpenConnection()
            r = db.get_by_codebar(row['Codigo'])
            if r:
                # print(r) 
                # pass
                aux = r
                aux.min_inventory = int(row['Inv. Minimo'])
                aux.retailsale = float(row['Precio Venta'])
                aux.wholesale = float(row['Precio Mayoreo'])
                if row['Inv. Minimo'] == 'nan':
                    print(row['Inv. Minimo'])
                else:
                    # print(f"No es nan {row['Inv. Minimo']}")
                    aux.inventory = row['Inv. Minimo']
                aux.lastUpdate = datetime.now
                db.update_product(row['Codigo'], aux) 
            else:
                p = ProductSchema(key=str(row['Codigo']), code=str(row['Codigo']), codebar=str(row['Codigo']),description=str(row['Descripcion']), buy=float(row['Precio Costo']), retailsale=float(row['Precio Venta']), wholesale=float(row['Precio Mayoreo']),min_inventory=0, inventory=int(row['Inv. Minimo']), department=str(row['Departamento']), lastUpdate= datetime.now(), codebarInner=str(f"{row['Codigo']}6"),codebarMaster=str(f"{row['Codigo']}12"), unit="",brand="",id=int(time.time()), box=int(0),master=int(0))
                db.create_product(obj_in=p)
            
            db.CloseConnection()
        except:           
            db.CloseConnection()
    print(f"----- Uploading Data to DB Finished ----- {time.time() - start_time} s")