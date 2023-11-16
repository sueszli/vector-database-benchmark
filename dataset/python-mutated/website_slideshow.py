import frappe
from frappe import _
from frappe.model.document import Document

class WebsiteSlideshow(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        from frappe.website.doctype.website_slideshow_item.website_slideshow_item import WebsiteSlideshowItem
        header: DF.HTMLEditor | None
        slideshow_items: DF.Table[WebsiteSlideshowItem]
        slideshow_name: DF.Data

    def validate(self):
        if False:
            print('Hello World!')
        self.validate_images()

    def on_update(self):
        if False:
            i = 10
            return i + 15
        from frappe.website.utils import clear_cache
        clear_cache()

    def validate_images(self):
        if False:
            while True:
                i = 10
        'atleast one image file should be public for slideshow'
        files = map(lambda row: row.image, self.slideshow_items)
        if files:
            result = frappe.get_all('File', filters={'file_url': ('in', list(files))}, fields='is_private')
            if any((file.is_private for file in result)):
                frappe.throw(_('All Images attached to Website Slideshow should be public'))

def get_slideshow(doc):
    if False:
        while True:
            i = 10
    if not doc.slideshow:
        return {}
    slideshow = frappe.get_doc('Website Slideshow', doc.slideshow)
    return {'slides': slideshow.get({'doctype': 'Website Slideshow Item'}), 'slideshow_header': slideshow.header or ''}