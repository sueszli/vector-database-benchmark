from visidata import VisiData, vd, TableSheet, asyncthread, ColumnItem, Column, ColumnAttr, Progress
import visidata.loaders.google
from urllib.parse import urlparse

@VisiData.api
def openurl_imap(vd, url, **kwargs):
    if False:
        print('Hello World!')
    url_parsed = urlparse(str(url))
    return ImapSheet(url_parsed.hostname, source=url_parsed, password=url_parsed.password)

class ImapSheet(TableSheet):
    columns = [ColumnItem('message-id'), ColumnItem('folder'), ColumnItem('Date'), ColumnItem('From'), ColumnItem('To'), ColumnItem('Subject'), ColumnAttr('defects'), Column('payload', getter=lambda c, r: r.get_payload()), Column('content_type', getter=lambda c, r: r.get_content_type())]
    nKeys = 1

    def iterload(self):
        if False:
            i = 10
            return i + 15
        import imaplib
        import email.parser
        m = imaplib.IMAP4_SSL(host=self.source.hostname)
        user = self.source.username
        if self.source.hostname == 'imap.gmail.com':
            credentials = vd.google_auth(scopes='https://mail.google.com/')
            header_template = 'user=%s\x01auth=Bearer %s\x01\x01'
            m.authenticate('XOAUTH2', lambda x: header_template % (user, credentials.token))
        else:
            if self.password is None:
                vd.error('no password given in url')
            m.login(user, self.source.password)
        (typ, folders) = m.list()
        for r in Progress(folders, gerund='downloading'):
            fname = r.decode('utf-8').split()[-1]
            try:
                m.select(fname)
                (typ, data) = m.search(None, 'ALL')
                for num in data[0].split():
                    (typ, msgbytes) = m.fetch(num, '(RFC822)')
                    if typ != 'OK':
                        vd.warning(typ, msgbytes)
                        continue
                    msg = email.message_from_bytes(msgbytes[0][1])
                    msg['folder'] = fname
                    yield msg
                m.close()
            except Exception:
                vd.exceptionCaught()
        m.logout()

    def addRow(self, row, **kwargs):
        if False:
            while True:
                i = 10
        if row.is_multipart():
            for p in row.get_payload():
                for hdr in 'message-id folder Date From To Subject'.split():
                    if hdr in row:
                        p[hdr] = row[hdr]
                self.addRow(p, **kwargs)
        else:
            super().addRow(row, **kwargs)