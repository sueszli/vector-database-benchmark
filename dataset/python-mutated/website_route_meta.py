from frappe.model.document import Document

class WebsiteRouteMeta(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        from frappe.website.doctype.website_meta_tag.website_meta_tag import WebsiteMetaTag
        meta_tags: DF.Table[WebsiteMetaTag]

    def autoname(self):
        if False:
            for i in range(10):
                print('nop')
        if self.name and self.name.startswith('/'):
            self.name = self.name[1:]