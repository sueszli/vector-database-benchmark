from wechatpy.client.api.base import BaseWeChatAPI

class WeChatFreePublish(BaseWeChatAPI):

    def submit(self, media_id: str) -> dict:
        if False:
            while True:
                i = 10
        '\n        发布接口\n\n        详情请参考：\n        https://developers.weixin.qq.com/doc/offiaccount/Publish/Publish.html\n\n        :param media_id: 要发布的草稿的media_id\n        :return: 返回的 JSON 数据包\n        '
        return self._post('freepublish/submit', data={'media_id': media_id})

    def get(self, publish_id: str) -> dict:
        if False:
            print('Hello World!')
        '\n        发布状态轮询接口\n        开发者可以尝试通过下面的发布状态轮询接口获知发布情况。\n\n        详情请参考：\n        https://developers.weixin.qq.com/doc/offiaccount/Publish/Get_status.html\n\n        :param publish_id: 发布任务id\n        :return: 返回的 JSON 数据包\n        '
        return self._post('freepublish/get', data={'publish_id': publish_id})

    def delete(self, article_id: str, index: int=0) -> dict:
        if False:
            return 10
        '\n        删除发布\n        发布成功之后，随时可以通过该接口删除。此操作不可逆，请谨慎操作。\n\n        详情请参考：\n        https://developers.weixin.qq.com/doc/offiaccount/Publish/Delete_posts.html\n\n        :param article_id: 成功发布时返回的 article_id\n        :param index: 要删除的文章在图文消息中的位置，第一篇编号为1，该字段不填或填 0 会删除全部文章\n        :return: 返回的 JSON 数据包\n        '
        return self._post('freepublish/delete', data={'article_id': article_id, 'index': index})

    def getarticle(self, article_id: str) -> dict:
        if False:
            i = 10
            return i + 15
        '\n        通过 article_id 获取已发布文章\n        开发者可以通过 article_id 获取已发布的图文信息。\n\n        详情请参考：\n        https://developers.weixin.qq.com/doc/offiaccount/Publish/Get_article_from_id.html\n\n        :param article_id: 要获取的草稿的article_id\n        :return: 返回的 JSON 数据包\n        '
        return self._post('freepublish/getarticle', data={'article_id': article_id})

    def batchget(self, offset: int, count: int, no_content: int=0) -> dict:
        if False:
            while True:
                i = 10
        '\n        获取成功发布列表\n        开发者可以获取已成功发布的消息列表。\n\n        详情请参考：\n        https://developers.weixin.qq.com/doc/offiaccount/Publish/Get_publication_records.html\n\n        :param offset: 从全部素材的该偏移位置开始返回，0表示从第一个素材返回\n        :param count: 返回素材的数量，取值在1到20之间\n        :param no_content: 1 表示不返回 content 字段，0 表示正常返回，默认为 0\n        :return: 返回的 JSON 数据包\n        '
        return self._post('freepublish/batchget', data={'offset': offset, 'count': count, 'no_content': no_content})