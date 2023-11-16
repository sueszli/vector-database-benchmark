import frappe
from frappe.tests.utils import FrappeTestCase

class TestHelpArticle(FrappeTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            for i in range(10):
                print('nop')
        cls.help_category = frappe.get_doc({'doctype': 'Help Category', 'category_name': '_Test Help Category'}).insert()
        cls.help_article = frappe.get_doc({'doctype': 'Help Article', 'title': '_Test Article', 'category': cls.help_category.name, 'content': '_Test Article'}).insert()

    def test_article_is_helpful(self):
        if False:
            while True:
                i = 10
        from frappe.website.doctype.help_article.help_article import add_feedback
        self.help_article.load_from_db()
        self.assertEqual(self.help_article.helpful, 0)
        self.assertEqual(self.help_article.not_helpful, 0)
        add_feedback(self.help_article.name, 'Yes')
        add_feedback(self.help_article.name, 'No')
        self.help_article.load_from_db()
        self.assertEqual(self.help_article.helpful, 1)
        self.assertEqual(self.help_article.not_helpful, 1)

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            return 10
        frappe.delete_doc(cls.help_article.doctype, cls.help_article.name)
        frappe.delete_doc(cls.help_category.doctype, cls.help_category.name)