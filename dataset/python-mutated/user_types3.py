import xlsxwriter

def hide_password(worksheet, row, col, string, format=None):
    if False:
        return 10
    if col == 1 and row > 0:
        return worksheet.write_string(row, col, '****', format)
    else:
        return worksheet.write_string(row, col, string, format)
workbook = xlsxwriter.Workbook('user_types3.xlsx')
worksheet = workbook.add_worksheet()
bold = workbook.add_format({'bold': True})
worksheet.set_row(0, None, bold)
worksheet.add_write_handler(str, hide_password)
my_data = [['Name', 'Password', 'City'], ['Sara', '$5%^6&', 'Rome'], ['Michele', '123abc', 'Milano'], ['Maria', 'juvexme', 'Torino'], ['Paolo', 'qwerty', 'Fano']]
for (row_num, row_data) in enumerate(my_data):
    worksheet.write_row(row_num, 0, row_data)
workbook.close()