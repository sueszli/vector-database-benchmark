import hashlib
import frappe
from frappe.model.document import Document
from frappe.query_builder import DocType
from frappe.utils import cint, now_datetime

class TransactionLog(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        amended_from: DF.Link | None
        chaining_hash: DF.SmallText | None
        checksum_version: DF.Data | None
        data: DF.LongText | None
        document_name: DF.Data | None
        previous_hash: DF.SmallText | None
        reference_doctype: DF.Data | None
        row_index: DF.Data | None
        timestamp: DF.Datetime | None
        transaction_hash: DF.SmallText | None

    def before_insert(self):
        if False:
            for i in range(10):
                print('nop')
        index = get_current_index()
        self.row_index = index
        self.timestamp = now_datetime()
        if index != 1:
            prev_hash = frappe.get_all('Transaction Log', filters={'row_index': str(index - 1)}, pluck='chaining_hash', limit=1)
            if prev_hash:
                self.previous_hash = prev_hash[0]
            else:
                self.previous_hash = 'Indexing broken'
        else:
            self.previous_hash = self.hash_line()
        self.transaction_hash = self.hash_line()
        self.chaining_hash = self.hash_chain()
        self.checksum_version = 'v1.0.1'

    def hash_line(self):
        if False:
            for i in range(10):
                print('nop')
        sha = hashlib.sha256()
        sha.update(frappe.safe_encode(str(self.row_index)) + frappe.safe_encode(str(self.timestamp)) + frappe.safe_encode(str(self.data)))
        return sha.hexdigest()

    def hash_chain(self):
        if False:
            print('Hello World!')
        sha = hashlib.sha256()
        sha.update(frappe.safe_encode(str(self.transaction_hash)) + frappe.safe_encode(str(self.previous_hash)))
        return sha.hexdigest()

def get_current_index():
    if False:
        while True:
            i = 10
    series = DocType('Series')
    current = frappe.qb.from_(series).where(series.name == 'TRANSACTLOG').for_update().select('current').run()
    if current and current[0][0] is not None:
        current = current[0][0]
        frappe.db.sql("UPDATE `tabSeries`\n\t\t\tSET `current` = `current` + 1\n\t\t\twhere `name` = 'TRANSACTLOG'")
        current = cint(current) + 1
    else:
        frappe.db.sql("INSERT INTO `tabSeries` (name, current) VALUES ('TRANSACTLOG', 1)")
        current = 1
    return current