class PagesDataStore(object):

    def __init__(self, db):
        if False:
            for i in range(10):
                print('nop')
        self.db = db
        pass

    def add_link_to_crawl(self, url):
        if False:
            for i in range(10):
                print('nop')
        'Add the given link to `links_to_crawl`.'
        pass

    def remove_link_to_crawl(self, url):
        if False:
            for i in range(10):
                print('nop')
        'Remove the given link from `links_to_crawl`.'
        pass

    def reduce_priority_link_to_crawl(self, url):
        if False:
            return 10
        'Reduce the priority of a link in `links_to_crawl` to avoid cycles.'
        pass

    def extract_max_priority_page(self):
        if False:
            return 10
        'Return the highest priority link in `links_to_crawl`.'
        pass

    def insert_crawled_link(self, url, signature):
        if False:
            return 10
        'Add the given link to `crawled_links`.'
        pass

    def crawled_similar(self, signature):
        if False:
            return 10
        "Determine if we've already crawled a page matching the given signature"
        pass

class Page(object):

    def __init__(self, url, contents, child_urls):
        if False:
            while True:
                i = 10
        self.url = url
        self.contents = contents
        self.child_urls = child_urls
        self.signature = self.create_signature()

    def create_signature(self):
        if False:
            print('Hello World!')
        pass

class Crawler(object):

    def __init__(self, pages, data_store, reverse_index_queue, doc_index_queue):
        if False:
            print('Hello World!')
        self.pages = pages
        self.data_store = data_store
        self.reverse_index_queue = reverse_index_queue
        self.doc_index_queue = doc_index_queue

    def crawl_page(self, page):
        if False:
            for i in range(10):
                print('nop')
        for url in page.child_urls:
            self.data_store.add_link_to_crawl(url)
        self.reverse_index_queue.generate(page)
        self.doc_index_queue.generate(page)
        self.data_store.remove_link_to_crawl(page.url)
        self.data_store.insert_crawled_link(page.url, page.signature)

    def crawl(self):
        if False:
            print('Hello World!')
        while True:
            page = self.data_store.extract_max_priority_page()
            if page is None:
                break
            if self.data_store.crawled_similar(page.signature):
                self.data_store.reduce_priority_link_to_crawl(page.url)
            else:
                self.crawl_page(page)
            page = self.data_store.extract_max_priority_page()