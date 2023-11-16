from whoosh.fields import ID, TEXT, Schema
from whoosh.index import EmptyIndexError, create_in, open_dir
from whoosh.qparser import FieldsPlugin, MultifieldParser, WildcardPlugin
from whoosh.query import FuzzyTerm, Prefix
from whoosh.writing import AsyncWriter
import frappe
from frappe.utils import update_progress_bar

class FullTextSearch:
    """Frappe Wrapper for Whoosh"""

    def __init__(self, index_name):
        if False:
            return 10
        self.index_name = index_name
        self.index_path = get_index_path(index_name)
        self.schema = self.get_schema()
        self.id = self.get_id()

    def get_schema(self):
        if False:
            for i in range(10):
                print('nop')
        return Schema(name=ID(stored=True), content=TEXT(stored=True))

    def get_fields_to_search(self):
        if False:
            print('Hello World!')
        return ['name', 'content']

    def get_id(self):
        if False:
            print('Hello World!')
        return 'name'

    def get_items_to_index(self):
        if False:
            for i in range(10):
                print('nop')
        'Get all documents to be indexed conforming to the schema'
        return []

    def get_document_to_index(self):
        if False:
            while True:
                i = 10
        return {}

    def build(self):
        if False:
            for i in range(10):
                print('nop')
        'Build search index for all documents'
        self.documents = self.get_items_to_index()
        self.build_index()

    def update_index_by_name(self, doc_name):
        if False:
            print('Hello World!')
        'Wraps `update_index` method, gets the document from name\n\t\tand updates the index. This function changes the current user\n\t\tand should only be run as administrator or in a background job.\n\n\t\tArgs:\n\t\t        self (object): FullTextSearch Instance\n\t\t        doc_name (str): name of the document to be updated\n\t\t'
        document = self.get_document_to_index(doc_name)
        if document:
            self.update_index(document)

    def remove_document_from_index(self, doc_name):
        if False:
            while True:
                i = 10
        'Remove document from search index\n\n\t\tArgs:\n\t\t        self (object): FullTextSearch Instance\n\t\t        doc_name (str): name of the document to be removed\n\t\t'
        if not doc_name:
            return
        ix = self.get_index()
        with ix.searcher():
            writer = AsyncWriter(ix)
            writer.delete_by_term(self.id, doc_name)
            writer.commit(optimize=True)

    def update_index(self, document):
        if False:
            print('Hello World!')
        'Update search index for a document\n\n\t\tArgs:\n\t\t        self (object): FullTextSearch Instance\n\t\t        document (_dict): A dictionary with title, path and content\n\t\t'
        ix = self.get_index()
        with ix.searcher():
            writer = AsyncWriter(ix)
            writer.delete_by_term(self.id, document[self.id])
            writer.add_document(**document)
            writer.commit(optimize=True)

    def get_index(self):
        if False:
            print('Hello World!')
        try:
            return open_dir(self.index_path)
        except EmptyIndexError:
            return self.create_index()

    def create_index(self):
        if False:
            print('Hello World!')
        frappe.create_folder(self.index_path)
        return create_in(self.index_path, self.schema)

    def build_index(self):
        if False:
            i = 10
            return i + 15
        'Build index for all parsed documents'
        ix = self.create_index()
        writer = AsyncWriter(ix)
        for (i, document) in enumerate(self.documents):
            if document:
                writer.add_document(**document)
            update_progress_bar('Building Index', i, len(self.documents))
        writer.commit(optimize=True)

    def search(self, text, scope=None, limit=20):
        if False:
            return 10
        'Search from the current index\n\n\t\tArgs:\n\t\t        text (str): String to search for\n\t\t        scope (str, optional): Scope to limit the search. Defaults to None.\n\t\t        limit (int, optional): Limit number of search results. Defaults to 20.\n\n\t\tReturns:\n\t\t        [List(_dict)]: Search results\n\t\t'
        ix = self.get_index()
        results = None
        search_fields = self.get_fields_to_search()
        fieldboosts = {}
        for (idx, field) in enumerate(search_fields, start=1):
            fieldboosts[field] = 1.0 / idx
        with ix.searcher() as searcher:
            parser = MultifieldParser(search_fields, ix.schema, termclass=FuzzyTermExtended, fieldboosts=fieldboosts)
            parser.remove_plugin_class(FieldsPlugin)
            parser.remove_plugin_class(WildcardPlugin)
            query = parser.parse(text)
            filter_scoped = None
            if scope:
                filter_scoped = Prefix(self.id, scope)
            results = searcher.search(query, limit=limit, filter=filter_scoped)
            return [self.parse_result(r) for r in results]

class FuzzyTermExtended(FuzzyTerm):

    def __init__(self, fieldname, text, boost=1.0, maxdist=2, prefixlength=1, constantscore=True):
        if False:
            i = 10
            return i + 15
        super().__init__(fieldname, text, boost=boost, maxdist=maxdist, prefixlength=prefixlength, constantscore=constantscore)

def get_index_path(index_name):
    if False:
        while True:
            i = 10
    return frappe.get_site_path('indexes', index_name)