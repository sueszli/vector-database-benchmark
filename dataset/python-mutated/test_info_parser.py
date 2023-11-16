from unittest.mock import patch
from .util import mock_request_get_content
from weibo_spider.parser.info_parser import InfoParser

@patch('requests.get', mock_request_get_content)
def test_info_parser():
    if False:
        for i in range(10):
            print('nop')
    info_parser = InfoParser(cookie='', user_id='1669879400')
    user = info_parser.extract_user_info()
    assert user.nickname == 'Dear-迪丽热巴'