import time
from datetime import datetime
import pandas as pd
import numpy
import random

from crud.crud_products import CRUDproductsObject



def addDataFromCSV():
    db = CRUDproductsObject

    db.OpenConnection()


    conn = db.conn
    cursor = db.cursor

    # cursor.execute("DELETE FROM product *")
    conn.commit()

    file_path = "assets\data\catalogo_utf8.csv"
    data = pd.read_csv(file_path, encoding='utf-8')


    start_time = time.time()
    print(f"----- Uploading Data to DB -----")
    size = len(data.index)
    values=[]
    for j in range(size):
        row = data.iloc[j]
        if row.isna().values.any():
            row.fillna(int(datetime.now().timestamp()))
        row = data.iloc[j].values.flatten().tolist()
        for i in range(len(row)):
            if(type(row[i]) == numpy.int64):
                row[i] = int(row[i])

            if(i in range(8,11)):
                if row[i] == '*':
                    row[i] = float(0)
                else:
                    row[i] = float(row[i])

            if(i in range(11,13)):
                if(type(row[i]) != int):
                    row[i] = int(0)

            if(i in range(2,5)):
                if(row[i] != row[i]):
                    row[i] = (row[i-1]+random.randint(3, 10000))
                row[i] = int(row[i])
            if(i==14):
                if(row[i] == row[i]):
                    row[i] = int(row[i])
                else:
                    row[i]=0
        row.append(datetime.now())
        values.append(tuple(row))

    args = ','.join(cursor.mogrify("(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", j).decode('utf-8')
                    for j in values)

    cursor.execute("INSERT INTO product VALUES " + (args))

    conn.commit()

    print(f"----- Uploading Data to DB Finished ----- {time.time() - start_time} s")
    db.CloseConnection()