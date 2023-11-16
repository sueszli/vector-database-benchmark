from frappe.website.utils import clear_cache
from frappe.website.website_generator import WebsiteGenerator

class BlogCategory(WebsiteGenerator):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        description: DF.SmallText | None
        preview_image: DF.AttachImage | None
        published: DF.Check
        route: DF.Data | None
        title: DF.Data

    def autoname(self):
        if False:
            while True:
                i = 10
        self.name = self.scrub(self.title)

    def on_update(self):
        if False:
            return 10
        clear_cache()

    def set_route(self):
        if False:
            print('Hello World!')
        self.route = 'blog/' + self.name