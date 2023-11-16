"""
@project: PyCharm
@file: excel_func.py
@author: Shengqiang Zhang
@time: 2020/4/11 21:14
@mail: sqzhang77@gmail.com
"""
import xlrd
import xlwt
from xlutils.copy import copy

def write_excel_xls(path, sheet_name_list, value):
    if False:
        print('Hello World!')
    workbook = xlwt.Workbook()
    index = len(value)
    for sheet_name in sheet_name_list:
        sheet = workbook.add_sheet(sheet_name)
        for i in range(0, index):
            for j in range(0, len(value[i])):
                sheet.write(i, j, value[i][j])
    workbook.save(path)

def write_excel_xls_append(path, sheet_name, value):
    if False:
        while True:
            i = 10
    index = len(value)
    workbook = xlrd.open_workbook(path)
    worksheet = workbook.sheet_by_name(sheet_name)
    rows_old = worksheet.nrows
    new_workbook = copy(workbook)
    new_worksheet = new_workbook.get_sheet(sheet_name)
    for i in range(0, index):
        for j in range(0, len(value[i])):
            new_worksheet.write(i + rows_old, j, value[i][j])
    new_workbook.save(path)
    print('{}【追加】写入【{}】数据成功！'.format(path, sheet_name))