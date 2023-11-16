"""Custom VisiData save format"""
import json
from visidata import VisiData, JsonSheet, Progress, IndexSheet, SettableColumn, ItemColumn, ExprColumn
NL = '\n'

@VisiData.api
def open_vds(vd, p):
    if False:
        for i in range(10):
            print('nop')
    return VdsIndexSheet(p.name, source=p)

@VisiData.api
def save_vds(vd, p, *sheets):
    if False:
        while True:
            i = 10
    'Save in custom VisiData format, preserving columns and their attributes.'
    with p.open(mode='w', encoding='utf-8') as fp:
        for vs in sheets:
            d = {'name': vs.name}
            fp.write('#' + json.dumps(d) + NL)
            for col in vs.columns:
                d = col.__getstate__()
                if isinstance(col, SettableColumn):
                    d['col'] = 'Column'
                elif isinstance(col, ItemColumn):
                    d['col'] = 'Column'
                    d['expr'] = col.name
                else:
                    d['col'] = type(col).__name__
                fp.write('#' + json.dumps(d) + NL)
            with Progress(gerund='saving'):
                for row in vs.iterdispvals(*vs.columns, format=False):
                    d = {col.name: val for (col, val) in row.items()}
                    fp.write(json.dumps(d, default=str) + NL)

class VdsIndexSheet(IndexSheet):

    def iterload(self):
        if False:
            i = 10
            return i + 15
        vs = None
        with self.source.open(encoding='utf-8') as fp:
            line = fp.readline()
            while line:
                if line.startswith('#{'):
                    d = json.loads(line[1:])
                    if 'col' not in d:
                        vs = VdsSheet(d.pop('name'), columns=[], source=self.source, source_fpos=fp.tell())
                        yield vs
                line = fp.readline()

class VdsSheet(JsonSheet):

    def newRow(self):
        if False:
            i = 10
            return i + 15
        return {}

    def iterload(self):
        if False:
            while True:
                i = 10
        self.colnames = {}
        self.columns = []
        with self.source.open(encoding='utf-8') as fp:
            fp.seek(self.source_fpos)
            line = fp.readline()
            while line and line.startswith('#{'):
                d = json.loads(line[1:])
                if 'col' not in d:
                    raise Exception(d)
                classname = d.pop('col')
                if classname == 'Column':
                    classname = 'ItemColumn'
                    d['expr'] = d['name']
                c = globals()[classname](d.pop('name'), sheet=self)
                self.addColumn(c)
                self.colnames[c.name] = c
                for (k, v) in d.items():
                    setattr(c, k, v)
                line = fp.readline()
            while line and (not line.startswith('#{')):
                d = json.loads(line)
                yield d
                line = fp.readline()