import os

class DataSet:

    def __init__(self, file, reader, formatter):
        if False:
            i = 10
            return i + 15
        self.reader = reader
        self.file = file
        self.formatter = formatter
        self.data = []

    def read(self):
        if False:
            i = 10
            return i + 15
        if os.getenv('DEBUG', '').lower() in ['1', 'y', 'yes', 't']:
            self.data.extend(filter(lambda record: record is not None, self.formatter.format(self.reader.read(self.file)[:10])))
        else:
            self.data.extend(filter(lambda record: record is not None, self.formatter.format(self.reader.read(self.file))))