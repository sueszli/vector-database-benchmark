import random
from locust import HttpUser, SequentialTaskSet, task, between
from pyquery import PyQuery

class BrowseDocumentationSequence(SequentialTaskSet):

    def on_start(self):
        if False:
            return 10
        self.urls_on_current_page = self.toc_urls = None

    @task
    def index_page(self):
        if False:
            i = 10
            return i + 15
        r = self.client.get('/')
        pq = PyQuery(r.content)
        link_elements = pq('.toctree-wrapper a.internal')
        self.toc_urls = [l.attrib['href'] for l in link_elements]
        self.client.get('/favicon.ico')

    @task
    def load_page(self, url=None):
        if False:
            for i in range(10):
                print('nop')
        url = random.choice(self.toc_urls)
        r = self.client.get(url)
        pq = PyQuery(r.content)
        link_elements = pq('a.internal')
        self.urls_on_current_page = [l.attrib['href'] for l in link_elements]

    @task
    def load_sub_page(self):
        if False:
            i = 10
            return i + 15
        url = random.choice(self.urls_on_current_page)
        r = self.client.get(url)

class AwesomeUser(HttpUser):
    tasks = [BrowseDocumentationSequence]
    host = 'https://docs.locust.io/en/latest/'
    wait_time = between(20, 600)