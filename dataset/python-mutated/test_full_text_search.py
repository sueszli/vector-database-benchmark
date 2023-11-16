from frappe.search.full_text_search import FullTextSearch
from frappe.tests.utils import FrappeTestCase

class TestFullTextSearch(FrappeTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        index = get_index()
        index.build()
        self.index = index

    def test_search_term(self):
        if False:
            while True:
                i = 10
        res = self.index.search('multilingual online encyclopedia')
        self.assertEqual(res[0], 'site/wikipedia')
        res = self.index.search('Linux kernel')
        self.assertEqual(res[0], 'os/linux')
        res = self.index.search('Enterprise Resource Planning')
        self.assertEqual(res[0], 'sw/erpnext')

    def test_search_limit(self):
        if False:
            i = 10
            return i + 15
        res = self.index.search('CommonSearchTerm')
        self.assertEqual(len(res), 5)
        res = self.index.search('CommonSearchTerm', limit=3)
        self.assertEqual(len(res), 3)
        res = self.index.search('CommonSearchTerm', limit=20)
        self.assertEqual(len(res), 5)

    def test_search_scope(self):
        if False:
            while True:
                i = 10
        res = self.index.search('multilingual online encyclopedia', scope=['os'])
        self.assertEqual(len(res), 0)
        res = self.index.search('CommonSearchTerm', scope=['os'])
        self.assertEqual(len(res), 2)
        self.assertTrue('os/linux' in res)
        self.assertTrue('os/gnu' in res)

    def test_remove_document_from_index(self):
        if False:
            return 10
        self.index.remove_document_from_index('os/gnu')
        res = self.index.search('GNU')
        self.assertEqual(len(res), 0)

    def test_update_index(self):
        if False:
            i = 10
            return i + 15
        self.index.update_index({'name': 'sw/erpnext', 'content': 'AwesomeERPNext'})
        res = self.index.search('CommonSearchTerm')
        self.assertTrue('sw/erpnext' not in res)
        res = self.index.search('AwesomeERPNext')
        self.assertEqual(res[0], 'sw/erpnext')
        self.index.update_index({'name': 'sw/frappebooks', 'content': 'DesktopAccounting'})
        res = self.index.search('DesktopAccounting')
        self.assertEqual(res[0], 'sw/frappebooks')

class TestWrapper(FullTextSearch):

    def get_items_to_index(self):
        if False:
            for i in range(10):
                print('nop')
        return get_documents()

    def get_document_to_index(self, name):
        if False:
            print('Hello World!')
        documents = get_documents()
        for doc in documents:
            if doc['name'] == name:
                return doc

    def parse_result(self, result):
        if False:
            return 10
        return result['name']

def get_index():
    if False:
        for i in range(10):
            print('nop')
    return TestWrapper('test_frappe_index')

def get_documents():
    if False:
        print('Hello World!')
    docs = []
    docs.append({'name': 'site/wikipedia', 'content': 'Wikipedia is a multilingual online encyclopedia created and maintained\n\t\t\tas an open collaboration project by a community of volunteer editors using a wiki-based editing system.\n\t\t\tIt is the largest and most popular general reference work on the World Wide Web. CommonSearchTerm'})
    docs.append({'name': 'os/linux', 'content': 'Linux is a family of open source Unix-like operating systems based on the\n\t\t\tLinux kernel, an operating system kernel first released on September 17, 1991, by Linus Torvalds.\n\t\t\tLinux is typically packaged in a Linux distribution. CommonSearchTerm'})
    docs.append({'name': 'os/gnu', 'content': 'GNU is an operating system and an extensive collection of computer software.\n\t\t\tGNU is composed wholly of free software, most of which is licensed under the GNU Project\'s own\n\t\t\tGeneral Public License. GNU is a recursive acronym for "GNU\'s Not Unix! ",\n\t\t\tchosen because GNU\'s design is Unix-like, but differs from Unix by being free software and containing no Unix code. CommonSearchTerm'})
    docs.append({'name': 'sw/erpnext', 'content': 'ERPNext is a free and open-source integrated Enterprise Resource Planning software developed by\n\t\t\tFrappe Technologies Pvt. Ltd. and is built on MariaDB database system using a Python based server-side framework.\n\t\t\tERPNext is a generic ERP software used by manufacturers, distributors and services companies. CommonSearchTerm'})
    docs.append({'name': 'sw/frappe', 'content': 'Frappe Framework is a full-stack web framework, that includes everything you need to build and\n\t\t\tdeploy business applications with Rich Admin Interface. CommonSearchTerm'})
    return docs