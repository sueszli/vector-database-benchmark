import csv
import json
from io import StringIO
import requests
import frappe
from frappe import _, msgprint
from frappe.utils import cint, comma_or, cstr, flt

def read_csv_content_from_attached_file(doc):
    if False:
        i = 10
        return i + 15
    fileid = frappe.get_all('File', fields=['name'], filters={'attached_to_doctype': doc.doctype, 'attached_to_name': doc.name}, order_by='creation desc')
    if fileid:
        fileid = fileid[0].name
    if not fileid:
        msgprint(_('File not attached'))
        raise Exception
    try:
        _file = frappe.get_doc('File', fileid)
        fcontent = _file.get_content()
        return read_csv_content(fcontent)
    except Exception:
        frappe.throw(_('Unable to open attached file. Did you export it as CSV?'), title=_('Invalid CSV Format'))

def read_csv_content(fcontent):
    if False:
        i = 10
        return i + 15
    if not isinstance(fcontent, str):
        decoded = False
        for encoding in ['utf-8', 'windows-1250', 'windows-1252']:
            try:
                fcontent = str(fcontent, encoding)
                decoded = True
                break
            except UnicodeDecodeError:
                continue
        if not decoded:
            frappe.msgprint(_('Unknown file encoding. Tried utf-8, windows-1250, windows-1252.'), raise_exception=True)
    fcontent = fcontent.encode('utf-8')
    content = [frappe.safe_decode(line) for line in fcontent.splitlines(True)]
    try:
        rows = []
        for row in csv.reader(content):
            r = []
            for val in row:
                val = val.strip()
                if val == '':
                    r.append(None)
                else:
                    r.append(val)
            rows.append(r)
        return rows
    except Exception:
        frappe.msgprint(_('Not a valid Comma Separated Value (CSV File)'))
        raise

@frappe.whitelist()
def send_csv_to_client(args):
    if False:
        while True:
            i = 10
    if isinstance(args, str):
        args = json.loads(args)
    args = frappe._dict(args)
    frappe.response['result'] = cstr(to_csv(args.data))
    frappe.response['doctype'] = args.filename
    frappe.response['type'] = 'csv'

def to_csv(data):
    if False:
        while True:
            i = 10
    writer = UnicodeWriter()
    for row in data:
        writer.writerow(row)
    return writer.getvalue()

def build_csv_response(data, filename):
    if False:
        for i in range(10):
            print('nop')
    frappe.response['result'] = cstr(to_csv(data))
    frappe.response['doctype'] = filename
    frappe.response['type'] = 'csv'

class UnicodeWriter:

    def __init__(self, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC):
        if False:
            print('Hello World!')
        self.encoding = encoding
        self.queue = StringIO()
        self.writer = csv.writer(self.queue, quoting=quoting)

    def writerow(self, row):
        if False:
            print('Hello World!')
        self.writer.writerow(row)

    def getvalue(self):
        if False:
            for i in range(10):
                print('nop')
        return self.queue.getvalue()

def check_record(d):
    if False:
        print('Hello World!')
    'check for mandatory, select options, dates. these should ideally be in doclist'
    from frappe.utils.dateutils import parse_date
    doc = frappe.get_doc(d)
    for key in d:
        docfield = doc.meta.get_field(key)
        val = d[key]
        if docfield:
            if docfield.reqd and (val == '' or val is None):
                frappe.msgprint(_('{0} is required').format(docfield.label), raise_exception=1)
            if docfield.fieldtype == 'Select' and val and docfield.options:
                if val not in docfield.options.split('\n'):
                    frappe.throw(_('{0} must be one of {1}').format(_(docfield.label), comma_or(docfield.options.split('\n'))))
            if val and docfield.fieldtype == 'Date':
                d[key] = parse_date(val)
            elif val and docfield.fieldtype in ['Int', 'Check']:
                d[key] = cint(val)
            elif val and docfield.fieldtype in ['Currency', 'Float', 'Percent']:
                d[key] = flt(val)

def import_doc(d, doctype, overwrite, row_idx, submit=False, ignore_links=False):
    if False:
        print('Hello World!')
    'import main (non child) document'
    if d.get('name') and frappe.db.exists(doctype, d['name']):
        if overwrite:
            doc = frappe.get_doc(doctype, d['name'])
            doc.flags.ignore_links = ignore_links
            doc.update(d)
            if d.get('docstatus') == 1:
                doc.update_after_submit()
            elif d.get('docstatus') == 0 and submit:
                doc.submit()
            else:
                doc.save()
            return 'Updated row (#%d) %s' % (row_idx + 1, getlink(doctype, d['name']))
        else:
            return 'Ignored row (#%d) %s (exists)' % (row_idx + 1, getlink(doctype, d['name']))
    else:
        doc = frappe.get_doc(d)
        doc.flags.ignore_links = ignore_links
        doc.insert()
        if submit:
            doc.submit()
        return 'Inserted row (#%d) %s' % (row_idx + 1, getlink(doctype, doc.get('name')))

def getlink(doctype, name):
    if False:
        print('Hello World!')
    return '<a href="/app/Form/%(doctype)s/%(name)s">%(name)s</a>' % locals()

def get_csv_content_from_google_sheets(url):
    if False:
        i = 10
        return i + 15
    validate_google_sheets_url(url)
    if 'gid=' in url:
        gid = url.rsplit('gid=', 1)[1]
    else:
        gid = 0
    url = url.rsplit('/edit', 1)[0]
    url = url + f'/export?format=csv&gid={gid}'
    headers = {'Accept': 'text/csv'}
    response = requests.get(url, headers=headers)
    if response.ok:
        if response.text.strip().endswith('</html>'):
            frappe.throw(_('Google Sheets URL is invalid or not publicly accessible.'), title=_('Invalid URL'))
        return response.content
    elif response.status_code == 400:
        frappe.throw(_('Google Sheets URL must end with "gid={number}". Copy and paste the URL from the browser address bar and try again.'), title=_('Incorrect URL'))
    else:
        response.raise_for_status()

def validate_google_sheets_url(url):
    if False:
        while True:
            i = 10
    from urllib.parse import urlparse
    u = urlparse(url)
    if u.scheme != 'https' or u.netloc != 'docs.google.com' or '/spreadsheets/' not in u.path:
        frappe.throw(_('"{0}" is not a valid Google Sheets URL').format(url), title=_('Invalid URL'))