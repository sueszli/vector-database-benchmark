import frappe
from frappe.website.doctype.help_article.help_article import clear_cache
from frappe.website.website_generator import WebsiteGenerator

class HelpCategory(WebsiteGenerator):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        category_description: DF.Text | None
        category_name: DF.Data
        help_articles: DF.Int
        published: DF.Check
        route: DF.Data | None
    website = frappe._dict(condition_field='published', page_title_field='category_name')

    def before_insert(self):
        if False:
            return 10
        self.published = 1

    def autoname(self):
        if False:
            for i in range(10):
                print('nop')
        self.name = self.category_name

    def validate(self):
        if False:
            i = 10
            return i + 15
        self.set_route()

    def set_route(self):
        if False:
            print('Hello World!')
        if not self.route:
            self.route = 'kb/' + self.scrub(self.category_name)

    def on_update(self):
        if False:
            print('Hello World!')
        clear_cache()