import xlsxwriter
import uuid

def write_uuid(worksheet, row, col, token, format=None):
    if False:
        while True:
            i = 10
    return worksheet.write_string(row, col, str(token), format)
workbook = xlsxwriter.Workbook('user_types1.xlsx')
worksheet = workbook.add_worksheet()
worksheet.set_column('A:A', 40)
worksheet.add_write_handler(uuid.UUID, write_uuid)
my_uuid = uuid.uuid3(uuid.NAMESPACE_DNS, 'python.org')
worksheet.write('A1', my_uuid)
workbook.close()