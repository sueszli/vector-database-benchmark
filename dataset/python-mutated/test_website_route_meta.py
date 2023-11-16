import frappe
from frappe.tests.utils import FrappeTestCase
from frappe.utils import set_request
from frappe.website.serve import get_response
test_dependencies = ['Blog Post']

class TestWebsiteRouteMeta(FrappeTestCase):

    def test_meta_tag_generation(self):
        if False:
            return 10
        blogs = frappe.get_all('Blog Post', fields=['name', 'route'], filters={'published': 1, 'route': ('!=', '')}, limit=1)
        blog = blogs[0]
        doc = frappe.new_doc('Website Route Meta')
        doc.append('meta_tags', {'key': 'type', 'value': 'blog_post'})
        doc.append('meta_tags', {'key': 'og:title', 'value': 'My Blog'})
        doc.name = blog.route
        doc.insert()
        set_request(path=blog.route)
        response = get_response()
        self.assertTrue(response.status_code, 200)
        html = self.normalize_html(response.get_data().decode())
        self.assertIn(self.normalize_html('<meta name="type" content="blog_post">'), html)
        self.assertIn(self.normalize_html('<meta property="og:title" content="My Blog">'), html)

    def tearDown(self):
        if False:
            while True:
                i = 10
        frappe.db.rollback()