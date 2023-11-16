import json
from visidata import VisiData, vd, SequenceSheet, deduceType, Progress

@VisiData.api
def guess_jsonla(vd, p):
    if False:
        while True:
            i = 10
    'A JSONLA file is a JSONL file with rows of arrays, where the first row\n    is a header array:\n\n    ["A", "B", "C"]\n    [1, "blue", true]\n    [2, "yellow", false]\n\n    The header array must be a flat array of strings\n\n    If no suitable header is found, fall back to generic JSON load.\n    '
    with p.open(encoding=vd.options.encoding) as fp:
        first_line = next(fp)
    if first_line.strip().startswith('['):
        ret = json.loads(first_line)
        if isinstance(ret, list) and all((isinstance(v, str) for v in ret)):
            return dict(filetype='jsonla')

@VisiData.api
def open_jsonla(vd, p):
    if False:
        while True:
            i = 10
    return JsonlArraySheet(p.name, source=p)

class JsonlArraySheet(SequenceSheet):
    rowtype = 'rows'

    def iterload(self):
        if False:
            i = 10
            return i + 15
        with self.open_text_source() as fp:
            for L in fp:
                yield json.loads(L)
        for (i, c) in enumerate(self.columns):
            c.type = deduceType(self.rows[0][i])

def get_jsonla_rows(sheet, cols):
    if False:
        return 10
    for row in Progress(sheet.rows):
        yield [col.getTypedValue(row) for col in cols]

class _vjsonEncoder(json.JSONEncoder):

    def default(self, obj):
        if False:
            for i in range(10):
                print('nop')
        return str(obj)

def write_jsonla(vs, fp):
    if False:
        return 10
    vcols = vs.visibleCols
    jsonenc = _vjsonEncoder()
    with Progress(gerund='saving'):
        header = [col.name for col in vcols]
        fp.write(jsonenc.encode(header) + '\n')
        rows = get_jsonla_rows(vs, vcols)
        for row in rows:
            fp.write(jsonenc.encode(row) + '\n')

@VisiData.api
def save_jsonla(vd, p, *vsheets):
    if False:
        while True:
            i = 10
    with p.open(mode='w', encoding=vsheets[0].options.save_encoding) as fp:
        for vs in vsheets:
            write_jsonla(vs, fp)
JsonlArraySheet.options.regex_skip = '^(//|#).*'