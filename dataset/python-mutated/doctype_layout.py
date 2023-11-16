from typing import TYPE_CHECKING
import frappe
from frappe.desk.utils import slug
from frappe.model.document import Document
if TYPE_CHECKING:
    from frappe.core.doctype.docfield.docfield import DocField

class DocTypeLayout(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.custom.doctype.doctype_layout_field.doctype_layout_field import DocTypeLayoutField
        from frappe.types import DF
        client_script: DF.Code | None
        document_type: DF.Link
        fields: DF.Table[DocTypeLayoutField]
        route: DF.Data

    def validate(self):
        if False:
            i = 10
            return i + 15
        if not self.route:
            self.route = slug(self.name)

    @frappe.whitelist()
    def sync_fields(self):
        if False:
            return 10
        doctype_fields = frappe.get_meta(self.document_type, cached=False).fields
        if self.is_new():
            added_fields = [field.fieldname for field in doctype_fields]
            removed_fields = []
        else:
            doctype_fieldnames = {field.fieldname for field in doctype_fields}
            layout_fieldnames = {field.fieldname for field in self.fields}
            added_fields = list(doctype_fieldnames - layout_fieldnames)
            removed_fields = list(layout_fieldnames - doctype_fieldnames)
        if not (added_fields or removed_fields):
            return
        added = self.add_fields(added_fields, doctype_fields)
        removed = self.remove_fields(removed_fields)
        for (index, field) in enumerate(self.fields):
            field.idx = index + 1
        return {'added': added, 'removed': removed}

    def add_fields(self, added_fields: list[str], doctype_fields: list['DocField']) -> list[dict]:
        if False:
            for i in range(10):
                print('nop')
        added = []
        for field in added_fields:
            field_details = next((f for f in doctype_fields if f.fieldname == field), None)
            if not field_details:
                continue
            row = self.append('fields', field_details.as_dict(no_default_fields=True))
            row_data = row.as_dict()
            if field_details.get('insert_after'):
                insert_after = next((f for f in self.fields if f.fieldname == field_details.insert_after), None)
                if insert_after:
                    self.fields.insert(insert_after.idx, row)
                    self.fields.pop()
                    row_data = {'idx': insert_after.idx + 1, 'fieldname': row.fieldname, 'label': row.label}
            added.append(row_data)
        return added

    def remove_fields(self, removed_fields: list[str]) -> list[dict]:
        if False:
            while True:
                i = 10
        removed = []
        for field in removed_fields:
            field_details = next((f for f in self.fields if f.fieldname == field), None)
            if field_details:
                self.remove(field_details)
                removed.append(field_details.as_dict())
        return removed