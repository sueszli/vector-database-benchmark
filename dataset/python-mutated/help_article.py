import frappe
from frappe import _
from frappe.rate_limiter import rate_limit
from frappe.utils import cint, is_markdown, markdown
from frappe.website.utils import get_comment_list
from frappe.website.website_generator import WebsiteGenerator

class HelpArticle(WebsiteGenerator):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        author: DF.Data | None
        category: DF.Link
        content: DF.TextEditor
        helpful: DF.Int
        level: DF.Literal['Beginner', 'Intermediate', 'Expert']
        likes: DF.Int
        not_helpful: DF.Int
        published: DF.Check
        route: DF.Data | None
        title: DF.Data

    def validate(self):
        if False:
            i = 10
            return i + 15
        self.set_route()

    def set_route(self):
        if False:
            i = 10
            return i + 15
        'Set route from category and title if missing'
        if not self.route:
            self.route = '/'.join([frappe.get_value('Help Category', self.category, 'route'), self.scrub(self.title)])

    def on_update(self):
        if False:
            for i in range(10):
                print('nop')
        self.update_category()
        clear_cache()

    def update_category(self):
        if False:
            while True:
                i = 10
        cnt = frappe.db.count('Help Article', filters={'category': self.category, 'published': 1})
        cat = frappe.get_doc('Help Category', self.category)
        cat.help_articles = cnt
        cat.save()

    def get_context(self, context):
        if False:
            print('Hello World!')
        if is_markdown(context.content):
            context.content = markdown(context.content)
        context.login_required = True
        context.category = frappe.get_doc('Help Category', self.category)
        context.level_class = get_level_class(self.level)
        context.comment_list = get_comment_list(self.doctype, self.name)
        context.show_sidebar = True
        context.sidebar_items = get_sidebar_items()
        context.parents = self.get_parents(context)

    def get_parents(self, context):
        if False:
            print('Hello World!')
        return [{'title': context.category.category_name, 'route': context.category.route}]

def get_list_context(context=None):
    if False:
        print('Hello World!')
    filters = dict(published=1)
    category = frappe.db.get_value('Help Category', {'route': frappe.local.path})
    if category:
        filters['category'] = category
    list_context = frappe._dict(title=category or _('Knowledge Base'), get_level_class=get_level_class, show_sidebar=True, sidebar_items=get_sidebar_items(), hide_filters=True, filters=filters, category=frappe.local.form_dict.category, no_breadcrumbs=True)
    if frappe.local.form_dict.txt:
        list_context.blog_subtitle = _('Filtered by "{0}"').format(frappe.local.form_dict.txt)
    return list_context

def get_level_class(level):
    if False:
        return 10
    return {'Beginner': 'green', 'Intermediate': 'orange', 'Expert': 'red'}[level]

def get_sidebar_items():
    if False:
        print('Hello World!')

    def _get():
        if False:
            for i in range(10):
                print('nop')
        return frappe.db.sql('select\n\t\t\t\tconcat(category_name, " (", help_articles, ")") as title,\n\t\t\t\tconcat(\'/\', route) as route\n\t\t\tfrom\n\t\t\t\t`tabHelp Category`\n\t\t\twhere\n\t\t\t\tpublished = 1 and help_articles > 0\n\t\t\torder by\n\t\t\t\thelp_articles desc', as_dict=True)
    return frappe.cache.get_value('knowledge_base:category_sidebar', _get)

def clear_cache():
    if False:
        i = 10
        return i + 15
    clear_website_cache()
    from frappe.website.utils import clear_cache
    clear_cache()

def clear_website_cache(path=None):
    if False:
        i = 10
        return i + 15
    frappe.cache.delete_value('knowledge_base:category_sidebar')
    frappe.cache.delete_value('knowledge_base:faq')

@frappe.whitelist(allow_guest=True)
@rate_limit(key='article', limit=5, seconds=60 * 60)
def add_feedback(article: str, helpful: str):
    if False:
        for i in range(10):
            print('nop')
    field = 'not_helpful' if helpful == 'No' else 'helpful'
    value = cint(frappe.db.get_value('Help Article', article, field))
    frappe.db.set_value('Help Article', article, field, value + 1, update_modified=False)