from xlsxwriter.workbook import Workbook
from xlsxwriter.worksheet import Worksheet
from xlsxwriter.worksheet import convert_cell_args

class MyWorksheet(Worksheet):
    """
    Subclass of the XlsxWriter Worksheet class to override the default
    write() method.

    """

    @convert_cell_args
    def write(self, row, col, *args):
        if False:
            for i in range(10):
                print('nop')
        data = args[0]
        if isinstance(data, str):
            data = data[::-1]
            return self.write_string(row, col, data)
        else:
            return super(MyWorksheet, self).write(row, col, *args)

class MyWorkbook(Workbook):
    """
    Subclass of the XlsxWriter Workbook class to override the default
    Worksheet class with our custom class.

    """

    def add_worksheet(self, name=None):
        if False:
            for i in range(10):
                print('nop')
        worksheet = super(MyWorkbook, self).add_worksheet(name, MyWorksheet)
        return worksheet
workbook = MyWorkbook('inheritance1.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write('A1', 'Hello')
worksheet.write('A2', 'World')
worksheet.write('A3', 123)
worksheet.write('A4', 345)
workbook.close()