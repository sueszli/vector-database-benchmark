import requests
from utils_cv.classification.data import Urls

def test_urls():
    if False:
        return 10
    all_urls = Urls.all()
    for url in all_urls:
        with requests.get(url):
            pass