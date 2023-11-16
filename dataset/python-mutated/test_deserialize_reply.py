import unittest
from wechatpy import replies

def create_reply(cls, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return cls(source='source', target='target', time=123456789, **kwargs)

class DeserializeReplyTestCase(unittest.TestCase):

    def test_empty_reply_deserialize(self):
        if False:
            print('Hello World!')
        reply = replies.EmptyReply()
        self._test_deserialize(reply)

    def test_text_reply_deserialize(self):
        if False:
            while True:
                i = 10
        reply = create_reply(replies.TextReply, content='test')
        self._test_deserialize(reply)

    def test_image_reply_deserialize(self):
        if False:
            i = 10
            return i + 15
        reply = create_reply(replies.ImageReply, media_id='media_id')
        self._test_deserialize(reply)

    def test_voice_reply_deserialize(self):
        if False:
            while True:
                i = 10
        reply = create_reply(replies.VoiceReply, media_id='media_id')
        self._test_deserialize(reply)

    def test_video_reply_deserialize(self):
        if False:
            for i in range(10):
                print('nop')
        reply = create_reply(replies.VideoReply, media_id='media_id', title='title', description='说个中文呗')
        self._test_deserialize(reply)
        reply = create_reply(replies.VideoReply, media_id='media_id', title='无描述')
        self._test_deserialize(reply)

    def test_music_reply_deserialize(self):
        if False:
            return 10
        reply = create_reply(replies.MusicReply, title='title', description='desc', music_url='music_url', hq_music_url='hq_music_url', thumb_media_id='thumb_media_id')
        self._test_deserialize(reply)
        reply = create_reply(replies.MusicReply, thumb_media_id='media_id')
        self._test_deserialize(reply)

    def test_article_reply_deserialize(self):
        if False:
            return 10
        articles = [{'title': 'test 1', 'description': 'test 1', 'image': 'http://www.qq.com/1.png', 'url': 'http://www.qq.com/1'}, {'title': 'test 2', 'description': 'test 2', 'image': 'http://www.qq.com/2.png', 'url': 'http://www.qq.com/2'}, {'title': 'test 3', 'description': 'test 3', 'image': 'http://www.qq.com/3.png', 'url': 'http://www.qq.com/3'}]
        reply = create_reply(replies.ArticlesReply, articles=articles)
        self._test_deserialize(reply)

    def _test_deserialize(self, reply):
        if False:
            return 10
        xml = reply.render()
        deserialized = replies.deserialize_reply(xml)
        xml2 = deserialized.render()
        self.assertEqual(xml, xml2)