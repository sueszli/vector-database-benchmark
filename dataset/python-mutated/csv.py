from visidata import vd, VisiData, SequenceSheet, options, stacktrace
from visidata import TypedExceptionWrapper, Progress
vd.option('csv_dialect', 'excel', 'dialect passed to csv.reader', replay=True)
vd.option('csv_delimiter', ',', 'delimiter passed to csv.reader', replay=True)
vd.option('csv_quotechar', '"', 'quotechar passed to csv.reader', replay=True)
vd.option('csv_skipinitialspace', True, 'skipinitialspace passed to csv.reader', replay=True)
vd.option('csv_escapechar', None, 'escapechar passed to csv.reader', replay=True)
vd.option('csv_lineterminator', '\r\n', 'lineterminator passed to csv.writer', replay=True)
vd.option('safety_first', False, 'sanitize input/output to handle edge cases, with a performance cost', replay=True)

@VisiData.api
def guess_csv(vd, p):
    if False:
        while True:
            i = 10
    import csv
    csv.field_size_limit(2 ** 31 - 1)
    line = next(p.open())
    if ',' in line:
        dialect = csv.Sniffer().sniff(line)
        r = dict(filetype='csv', _likelihood=0)
        for csvopt in dir(dialect):
            if not csvopt.startswith('_'):
                r['csv_' + csvopt] = getattr(dialect, csvopt)
        return r

@VisiData.api
def open_csv(vd, p):
    if False:
        i = 10
        return i + 15
    return CsvSheet(p.name, source=p)

def removeNulls(fp):
    if False:
        while True:
            i = 10
    for line in fp:
        yield line.replace('\x00', '')

class CsvSheet(SequenceSheet):
    _rowtype = list

    def iterload(self):
        if False:
            while True:
                i = 10
        'Convert from CSV, first handling header row specially.'
        import csv
        csv.field_size_limit(2 ** 31 - 1)
        with self.open_text_source() as fp:
            if options.safety_first:
                rdr = csv.reader(removeNulls(fp), **options.getall('csv_'))
            else:
                rdr = csv.reader(fp, **options.getall('csv_'))
            while True:
                try:
                    yield next(rdr)
                except csv.Error as e:
                    e.stacktrace = stacktrace()
                    yield [TypedExceptionWrapper(None, exception=e)]
                except StopIteration:
                    return

@VisiData.api
def save_csv(vd, p, sheet):
    if False:
        return 10
    'Save as single CSV file, handling column names as first line.'
    import csv
    csv.field_size_limit(2 ** 31 - 1)
    with p.open(mode='w', encoding=sheet.options.save_encoding, newline='') as fp:
        cw = csv.writer(fp, **options.getall('csv_'))
        colnames = [col.name for col in sheet.visibleCols]
        if ''.join(colnames):
            cw.writerow(colnames)
        with Progress(gerund='saving'):
            for dispvals in sheet.iterdispvals(format=True):
                cw.writerow(dispvals.values())
CsvSheet.options.regex_skip = '^#.*'
vd.addGlobals({'CsvSheet': CsvSheet})