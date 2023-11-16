from visidata import VisiData, Sheet, ItemColumn, date, Column

@VisiData.api
def open_mbox(vd, p):
    if False:
        while True:
            i = 10
    return MboxSheet(p.name, source=p, format='mbox')

@VisiData.api
def open_maildir(vd, p):
    if False:
        while True:
            i = 10
    return MboxSheet(p.name, source=p, format='Maildir')

@VisiData.api
def open_mmdf(vd, p):
    if False:
        while True:
            i = 10
    return MboxSheet(p.name, source=p, format='MMDF')

@VisiData.api
def open_babyl(vd, p):
    if False:
        i = 10
        return i + 15
    return MboxSheet(p.name, source=p, format='Babyl')

@VisiData.api
def open_mh(vd, p):
    if False:
        i = 10
        return i + 15
    return MboxSheet(p.name, source=p, format='MH')

class MboxSheet(Sheet):
    columns = [ItemColumn('Date', type=date), ItemColumn('From'), ItemColumn('To'), ItemColumn('Cc'), ItemColumn('Subject'), Column('Payload', getter=lambda c, r: r.get_payload(decode=True), setter=lambda c, r, v: r.set_payload(v))]

    def iterload(self):
        if False:
            return 10
        import mailbox
        cls = getattr(mailbox, self.format)
        self.mailbox = cls(str(self.source), create=False)
        for r in self.mailbox.itervalues():
            yield r