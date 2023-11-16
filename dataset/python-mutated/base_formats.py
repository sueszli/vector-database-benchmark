import csv
from io import BytesIO, StringIO
import openpyxl

class Dataset(list):

    def __init__(self, rows=(), headers=None):
        if False:
            i = 10
            return i + 15
        super().__init__(rows)
        self.headers = headers or (self.pop(0) if len(self) > 0 else [])

    def __str__(self):
        if False:
            print('Hello World!')
        'Print a table'
        result = [[]]
        widths = []
        for col in self.headers:
            value = str(col) if col is not None else ''
            result[0].append(value)
            widths.append(len(value) + 1)
        for row in self:
            result.append([])
            for (idx, col) in enumerate(row):
                value = str(col) if col is not None else ''
                result[-1].append(value)
                widths[idx] = max(widths[idx], len(value) + 1)
        row_formatter = '| '.join((f'{{{idx}:{width}}}' for (idx, width) in enumerate(widths))).format
        return '\n'.join((row_formatter(*row) for row in result))

class CSV:

    def is_binary(self):
        if False:
            return 10
        return False

    def get_read_mode(self):
        if False:
            return 10
        return 'r'

    def create_dataset(self, data, delimiter=','):
        if False:
            while True:
                i = 10
        '\n        Create dataset from csv data.\n        '
        return Dataset(csv.reader(StringIO(data), delimiter=delimiter))

class TSV(CSV):

    def create_dataset(self, data):
        if False:
            return 10
        '\n        Create dataset from tsv data.\n        '
        return super().create_dataset(data, delimiter='\t')

class XLSX:

    def is_binary(self):
        if False:
            print('Hello World!')
        return True

    def get_read_mode(self):
        if False:
            while True:
                i = 10
        return 'rb'

    def create_dataset(self, data):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create dataset from the first sheet of a xlsx workbook.\n        '
        workbook = openpyxl.load_workbook(BytesIO(data), read_only=True, data_only=True)
        sheet = workbook.worksheets[0]
        try:
            return Dataset((tuple((cell.value for cell in row)) for row in sheet.rows))
        finally:
            workbook.close()
DEFAULT_FORMATS = [CSV, XLSX, TSV]