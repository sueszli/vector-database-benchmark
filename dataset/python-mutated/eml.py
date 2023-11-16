import os
from visidata import VisiData, vd, Column, TableSheet, vlen

@VisiData.api
def open_eml(vd, p):
    if False:
        while True:
            i = 10
    return EmailSheet(p.name, source=p)

class EmailSheet(TableSheet):
    rowtype = 'parts'
    columns = [Column('filename', getter=lambda c, r: r.get_filename()), Column('content_type', getter=lambda c, r: r.get_content_type()), Column('payload', type=vlen, getter=lambda c, r: r.get_payload(decode=False))]

    def iterload(self):
        if False:
            while True:
                i = 10
        import email
        parser = email.parser.Parser()
        with self.source.open(encoding='utf-8') as fp:
            yield from parser.parse(fp).walk()

@EmailSheet.api
def extract_part(sheet, givenpath, part):
    if False:
        i = 10
        return i + 15
    with givenpath.open_bytes(mode='w') as fp:
        fp.write(part.get_payload(decode=True))

@EmailSheet.api
def extract_parts(sheet, givenpath, *parts):
    if False:
        print('Hello World!')
    'Save all *parts* to Path *givenpath*.'
    vd.confirmOverwrite(givenpath, f'{givenpath} already exists, extract anyway?')
    vd.status('saving %s parts to %s' % (len(parts), givenpath.given))
    if givenpath.is_dir() or givenpath.given.endswith('/') or len(parts) > 1:
        try:
            os.makedirs(givenpath, exist_ok=True)
        except FileExistsError:
            pass
        for part in parts:
            vd.execAsync(sheet.extract_part, givenpath / part.get_filename(), part)
    elif len(parts) == 1:
        vd.execAsync(sheet.extract_part, givenpath, parts[0])
    else:
        vd.fail('cannot save multiple parts to non-dir')
EmailSheet.addCommand('x', 'extract-part', 'extract_part(inputPath("save part as: ", value=cursorRow.get_filename()), cursorRow)')
EmailSheet.addCommand('gx', 'extract-part-selected', 'extract_parts(inputPath("save %d parts in: " % nSelectedRows), *selectedRows)')