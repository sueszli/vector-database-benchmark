import frappe
from frappe.model.document import Document

class WebsiteMetaTag(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        key: DF.Data
        parent: DF.Data
        parentfield: DF.Data
        parenttype: DF.Data
        value: DF.Text

    def get_content(self):
        if False:
            print('Hello World!')
        return (self.value or '').replace('\n', ' ')

    def get_meta_dict(self):
        if False:
            return 10
        return {self.key: self.get_content()}

    def set_in_context(self, context):
        if False:
            for i in range(10):
                print('nop')
        context.setdefault('metatags', frappe._dict({}))
        context.metatags[self.key] = self.get_content()
        return context