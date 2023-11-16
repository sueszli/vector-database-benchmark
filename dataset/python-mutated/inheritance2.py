from xlsxwriter.workbook import Workbook
from xlsxwriter.worksheet import Worksheet
from xlsxwriter.worksheet import convert_cell_args

def excel_string_width(str):
    if False:
        while True:
            i = 10
    "\n    Calculate the length of the string in Excel character units. This is only\n    an example and won't give accurate results. It will need to be replaced\n    by something more rigorous.\n\n    "
    string_width = len(str)
    if string_width == 0:
        return 0
    else:
        return string_width * 1.1

class MyWorksheet(Worksheet):
    """
    Subclass of the XlsxWriter Worksheet class to override the default
    write_string() method.

    """

    @convert_cell_args
    def write_string(self, row, col, string, cell_format=None):
        if False:
            return 10
        if self._check_dimensions(row, col):
            return -1
        min_width = 0
        string_width = excel_string_width(string)
        if string_width > min_width:
            max_width = self.max_column_widths.get(col, min_width)
            if string_width > max_width:
                self.max_column_widths[col] = string_width
        return super(MyWorksheet, self).write_string(row, col, string, cell_format)

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
        worksheet.max_column_widths = {}
        return worksheet

    def close(self):
        if False:
            return 10
        for worksheet in self.worksheets():
            for (column, width) in worksheet.max_column_widths.items():
                worksheet.set_column(column, column, width)
        return super(MyWorkbook, self).close()
workbook = MyWorkbook('inheritance2.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write('A1', 'F')
worksheet.write('B3', 'Foo')
worksheet.write('C1', 'F')
worksheet.write('C2', 'Fo')
worksheet.write('C3', 'Foo')
worksheet.write('C4', 'Food')
worksheet.write('D1', 'This is a longer string')
worksheet.write(0, 4, 'Hello World')
worksheet.write(0, 5, 123456)
workbook.close()