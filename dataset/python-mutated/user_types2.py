import xlsxwriter
import math

def ignore_nan(worksheet, row, col, number, format=None):
    if False:
        i = 10
        return i + 15
    if math.isnan(number):
        return worksheet.write_blank(row, col, None, format)
    else:
        return None
workbook = xlsxwriter.Workbook('user_types2.xlsx')
worksheet = workbook.add_worksheet()
worksheet.add_write_handler(float, ignore_nan)
my_data = [1, 2, float('nan'), 4, 5]
worksheet.write_row('A1', my_data)
workbook.close()