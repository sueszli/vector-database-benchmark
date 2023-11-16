from unittest.mock import patch
from .util import mock_request_get_content
from weibo_spider.parser.mblog_picAll_parser import MblogPicAllParser

@patch('requests.get', mock_request_get_content)
def test_mblog_picAll_parser():
    if False:
        i = 10
        return i + 15
    mblog_picAll_parser = MblogPicAllParser(cookie='', weibo_id='J5ZcSnCAg')
    preview_picture_list = mblog_picAll_parser.extract_preview_picture_list()
    assert len(preview_picture_list) == 18
    assert preview_picture_list[0] == 'http://ww3.sinaimg.cn/thumb180/63885668ly1gfn5qz5m1yj20u0140472.jpg'