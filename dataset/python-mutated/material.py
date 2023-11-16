import json
from wechatpy.client.api.base import BaseWeChatAPI

class WeChatMaterial(BaseWeChatAPI):

    def add(self, media_type, media_file, title=None, introduction=None):
        if False:
            print('Hello World!')
        '\n        新增其它类型永久素材\n        详情请参考\n        https://developers.weixin.qq.com/doc/offiaccount/Asset_Management/Adding_Permanent_Assets.html\n\n        :param media_type: 媒体文件类型，分别有图片（image）、语音（voice）、视频（video）和缩略图（thumb）\n        :param media_file: 要上传的文件，一个 File-object\n        :param title: 视频素材标题，仅上传视频素材时需要\n        :param introduction: 视频素材简介，仅上传视频素材时需要\n        :return: 返回的 JSON 数据包\n        '
        params = {'access_token': self.access_token, 'type': media_type}
        if media_type == 'video':
            assert title, 'Video title must be set'
            assert introduction, 'Video introduction must be set'
            description = {'title': title, 'introduction': introduction}
            params['description'] = json.dumps(description)
        return self._post('material/add_material', params=params, files={'media': media_file})

    def get(self, media_id):
        if False:
            return 10
        '\n        获取永久素材\n        详情请参考\n        https://developers.weixin.qq.com/doc/offiaccount/Asset_Management/Getting_Permanent_Assets.html\n\n        :param media_id: 素材的 media_id\n        :return: 图文素材返回图文列表，其它类型为素材的内容\n        '

        def _processor(res):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(res, dict) and 'news_item' in res:
                return res['news_item']
            return res
        return self._post('material/get_material', data={'media_id': media_id}, result_processor=_processor)

    def delete(self, media_id):
        if False:
            print('Hello World!')
        '\n        删除永久素材\n        详情请参考\n        https://developers.weixin.qq.com/doc/offiaccount/Asset_Management/Deleting_Permanent_Assets.html\n\n        :param media_id: 素材的 media_id\n        :return: 返回的 JSON 数据包\n        '
        return self._post('material/del_material', data={'media_id': media_id})

    def batchget(self, media_type, offset=0, count=20):
        if False:
            i = 10
            return i + 15
        '\n        获取素材列表\n        详情请参考\n        https://developers.weixin.qq.com/doc/offiaccount/Asset_Management/Get_materials_list.html\n\n        :param media_type: 媒体文件类型，分别有图片（image）、语音（voice）、视频（video）和缩略图（news）\n        :param offset: 从全部素材的该偏移位置开始返回，0 表示从第一个素材返回\n        :param count: 返回素材的数量，取值在1到20之间\n        :return: 返回的 JSON 数据包\n        '
        return self._post('material/batchget_material', data={'type': media_type, 'offset': offset, 'count': count})

    def get_count(self):
        if False:
            return 10
        '\n        获取素材总数\n        详情请参考\n        https://developers.weixin.qq.com/doc/offiaccount/Asset_Management/Get_the_total_of_all_materials.html\n\n        :return: 返回的 JSON 数据包\n        '
        return self._get('material/get_materialcount')

    def open_comment(self, msg_data_id, index=1):
        if False:
            print('Hello World!')
        '\n        打开已群发文章评论\n        https://mp.weixin.qq.com/wiki?id=mp1494572718_WzHIY\n        '
        return self._post('comment/open', data={'msg_data_id': msg_data_id, 'index': index})

    def close_comment(self, msg_data_id, index=1):
        if False:
            print('Hello World!')
        '\n        关闭已群发文章评论\n        '
        return self._post('comment/close', data={'msg_data_id': msg_data_id, 'index': index})

    def list_comment(self, msg_data_id, index=1, begin=0, count=50, type=0):
        if False:
            i = 10
            return i + 15
        '\n        查看指定文章的评论数据\n        '
        return self._post('comment/list', data={'msg_data_id': msg_data_id, 'index': index, 'begin': begin, 'count': count, 'type': type})

    def markelect_comment(self, msg_data_id, index, user_comment_id):
        if False:
            print('Hello World!')
        '\n        将评论标记精选\n        '
        return self._post('comment/markelect', data={'msg_data_id': msg_data_id, 'index': index, 'user_comment_id': user_comment_id})

    def unmarkelect_comment(self, msg_data_id, index, user_comment_id):
        if False:
            return 10
        '\n        将评论取消精选\n        '
        return self._post('comment/unmarkelect', data={'msg_data_id': msg_data_id, 'index': index, 'user_comment_id': user_comment_id})

    def delete_comment(self, msg_data_id, index, user_comment_id):
        if False:
            while True:
                i = 10
        '\n        删除评论\n        '
        return self._post('comment/delete', data={'msg_data_id': msg_data_id, 'index': index, 'user_comment_id': user_comment_id})

    def add_reply_comment(self, msg_data_id, index, user_comment_id, content):
        if False:
            i = 10
            return i + 15
        '\n        回复评论\n        '
        return self._post('comment/reply/add', data={'msg_data_id': msg_data_id, 'index': index, 'user_comment_id': user_comment_id, 'content': content})

    def delete_reply_comment(self, msg_data_id, index, user_comment_id):
        if False:
            for i in range(10):
                print('nop')
        '\n        删除回复\n        '
        return self._post('comment/reply/delete', data={'msg_data_id': msg_data_id, 'index': index, 'user_comment_id': user_comment_id})