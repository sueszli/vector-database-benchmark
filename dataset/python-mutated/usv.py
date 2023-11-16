from copy import copy
from visidata import Sheet, TsvSheet, options, vd, VisiData

@VisiData.api
def open_usv(vd, p):
    if False:
        for i in range(10):
            print('nop')
    return TsvSheet(p.name, source=p, delimiter='␟', row_delimiter='␞')

@VisiData.api
def save_usv(vd, p, vs):
    if False:
        for i in range(10):
            print('nop')
    vd.save_tsv(p, vs, row_delimiter='␞', delimiter='␟')