from visidata import VisiData, vd, Progress, IndexSheet

@VisiData.api
def open_frictionless(vd, p):
    if False:
        for i in range(10):
            print('nop')
    return FrictionlessIndexSheet(p.name, source=p)

class FrictionlessIndexSheet(IndexSheet):

    def iterload(self):
        if False:
            while True:
                i = 10
        datapackage = vd.importExternal('datapackage')
        self.dp = datapackage.Package(self.source.open(encoding='utf-8'))
        for r in Progress(self.dp.resources):
            yield vd.openSource(self.source.with_name(r.descriptor['path']), filetype=r.descriptor.get('format', 'json'))