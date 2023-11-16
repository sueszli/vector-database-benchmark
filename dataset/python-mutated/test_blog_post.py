import re
from bs4 import BeautifulSoup
import frappe
from frappe.custom.doctype.customize_form.customize_form import reset_customization
from frappe.tests.utils import FrappeTestCase
from frappe.utils import random_string, set_request
from frappe.website.doctype.blog_post.blog_post import get_blog_list
from frappe.website.serve import get_response
from frappe.website.utils import clear_website_cache
from frappe.website.website_generator import WebsiteGenerator
test_dependencies = ['Blog Post']

class TestBlogPost(FrappeTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        reset_customization('Blog Post')

    def tearDown(self):
        if False:
            return 10
        if hasattr(frappe.local, 'request'):
            delattr(frappe.local, 'request')

    def test_generator_view(self):
        if False:
            while True:
                i = 10
        pages = frappe.get_all('Blog Post', fields=['name', 'route'], filters={'published': 1, 'route': ('!=', '')}, limit=1)
        set_request(path=pages[0].route)
        response = get_response()
        self.assertTrue(response.status_code, 200)
        html = response.get_data().decode()
        self.assertTrue('<article class="blog-content" itemscope itemtype="http://schema.org/BlogPosting">' in html)

    def test_generator_not_found(self):
        if False:
            return 10
        pages = frappe.get_all('Blog Post', fields=['name', 'route'], filters={'published': 0}, limit=1)
        route = f'test-route-{frappe.generate_hash(length=5)}'
        frappe.db.set_value('Blog Post', pages[0].name, 'route', route)
        set_request(path=route)
        response = get_response()
        self.assertTrue(response.status_code, 404)

    def test_category_link(self):
        if False:
            for i in range(10):
                print('nop')
        blog = make_test_blog('Test Category Link')
        set_request(path=blog.route)
        blog_page_response = get_response()
        blog_page_html = frappe.safe_decode(blog_page_response.get_data())
        soup = BeautifulSoup(blog_page_html, 'html.parser')
        category_page_link = list(soup.find_all('a', href=re.compile(blog.blog_category)))[0]
        category_page_url = category_page_link['href']
        cached_value = frappe.db.value_cache.get(('DocType', 'Blog Post', 'name'))
        frappe.db.value_cache['DocType', 'Blog Post', 'name'] = (('Blog Post',),)
        set_request(path=category_page_url)
        category_page_response = get_response()
        category_page_html = frappe.safe_decode(category_page_response.get_data())
        self.assertIn(blog.title, category_page_html)
        frappe.db.value_cache['DocType', 'Blog Post', 'name'] = cached_value
        frappe.delete_doc('Blog Post', blog.name)
        frappe.delete_doc('Blog Category', blog.blog_category)

    def test_blog_pagination(self):
        if False:
            return 10
        (category_title, blogs, BLOG_COUNT) = ('List Category', [], 4)
        for index in range(BLOG_COUNT):
            blog = make_test_blog(category_title)
            blogs.append(blog)
        filters = frappe._dict({'blog_category': scrub(category_title)})
        self.assertEqual(len(get_blog_list(None, None, filters, 0, 3)), 3)
        self.assertEqual(len(get_blog_list(None, None, filters, 0, BLOG_COUNT)), BLOG_COUNT)
        self.assertEqual(len(get_blog_list(None, None, filters, 0, 2)), 2)
        self.assertEqual(len(get_blog_list(None, None, filters, 2, BLOG_COUNT)), 2)
        for blog in blogs:
            frappe.delete_doc(blog.doctype, blog.name)
        frappe.delete_doc('Blog Category', blogs[0].blog_category)

    def test_caching(self):
        if False:
            i = 10
            return i + 15
        frappe.flags.force_website_cache = True
        print(frappe.session.user)
        clear_website_cache()
        pages = frappe.get_all('Blog Post', fields=['name', 'route'], filters={'published': 1, 'title': '_Test Blog Post'}, limit=1)
        route = pages[0].route
        set_request(path=route)
        response = get_response()
        set_request(path=route)
        response = get_response()
        self.assertIn(('X-From-Cache', 'True'), list(response.headers))
        frappe.flags.force_website_cache = True

    def test_spam_comments(self):
        if False:
            print('Hello World!')
        blog = make_test_blog('Test Spam Comment')
        frappe.get_doc(doctype='Comment', comment_type='Comment', reference_doctype='Blog Post', reference_name=blog.name, comment_email='<a href="https://example.com/spam/">spam</a>', comment_by='<a href="https://example.com/spam/">spam</a>', published=1, content='More spam content. <a href="https://example.com/spam/">spam</a> with link.').insert()
        set_request(path=blog.route)
        blog_page_response = get_response()
        blog_page_html = frappe.safe_decode(blog_page_response.get_data())
        self.assertNotIn('<a href="https://example.com/spam/">spam</a>', blog_page_html)
        self.assertIn('More spam content. spam with link.', blog_page_html)
        frappe.delete_doc('Blog Post', blog.name)
        frappe.delete_doc('Blog Category', blog.blog_category)

    def test_like_dislike(self):
        if False:
            i = 10
            return i + 15
        test_blog = make_test_blog()
        frappe.db.delete('Comment', {'comment_type': 'Like', 'reference_doctype': 'Blog Post'})
        from frappe.templates.includes.likes.likes import like
        liked = like('Blog Post', test_blog.name, True)
        self.assertEqual(liked, True)
        disliked = like('Blog Post', test_blog.name, False)
        self.assertEqual(disliked, False)
        frappe.db.delete('Comment', {'comment_type': 'Like', 'reference_doctype': 'Blog Post'})
        test_blog.delete()

def scrub(text):
    if False:
        print('Hello World!')
    return WebsiteGenerator.scrub(None, text)

def make_test_blog(category_title='Test Blog Category'):
    if False:
        return 10
    category_name = scrub(category_title)
    if not frappe.db.exists('Blog Category', category_name):
        frappe.get_doc(dict(doctype='Blog Category', title=category_title)).insert()
    if not frappe.db.exists('Blogger', 'test-blogger'):
        frappe.get_doc(dict(doctype='Blogger', short_name='test-blogger', full_name='Test Blogger')).insert()
    return frappe.get_doc(dict(doctype='Blog Post', blog_category=category_name, blogger='test-blogger', title=random_string(20), route=random_string(20), content=random_string(20), published=1)).insert()