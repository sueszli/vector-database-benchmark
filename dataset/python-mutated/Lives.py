"""
@Description:Live.py
@Date       :2022/09/15 16:48:34
@Author     :JohnserfSeed
@version    :1.0
@License    :MIT License
@Github     :https://github.com/johnserf-seed
@Mail       :johnserfseed@gmail.com
-------------------------------------------------
Change Log  :
-------------------------------------------------
"""
import Util
XB = Util.XBogus()
URLS = Util.Urls()

class Lives:

    def __init__(self, cmd):
        if False:
            while True:
                i = 10
        self.headers = cmd.dyheaders

    def get_Live(self, live_url: str) -> None:
        if False:
            print('Hello World!')
        '\n        获取直播信息\n\n        Args:\n            live_url (str): 直播间链接\n\n        Returns:\n            live_json (dict): 直播间信息\n        '
        pattern = 'https://live\\.douyin\\.com/(\\d+)'
        match = Util.re.search(pattern, live_url)
        if match:
            web_rid = match.group(1)
        try:
            live_api = f"{URLS.LIVE}{XB.getXBogus(f'aid=6383&device_platform=web&web_rid={web_rid}')[0]}"
        except IndexError:
            raise Exception('检查是否为直播链接\r')
        response = Util.requests.request('GET', live_api, headers=self.headers)
        if response.text == '':
            input('[   🎦   ]:获取直播信息失败，请重新扫码登录\r')
            exit(0)
        live_json = response.json()
        api_status_code = live_json.get('status_code')
        if api_status_code == 4001038:
            input('[   📺   ]:该内容暂时无法无法查看，按回车退出')
            exit(0)
        live_status = live_json.get('data').get('data')[0].get('status')
        if live_status == 4:
            input('[   📺   ]:当前直播已结束，按回车退出')
            exit(0)
        title = live_json.get('data').get('data')[0].get('title')
        user_count = live_json.get('data').get('data')[0].get('user_count_str')
        nickname = Util.replaceT(live_json.get('data').get('data')[0].get('owner').get('nickname'))
        display_long = live_json.get('data').get('data')[0].get('room_view_stats').get('display_long')
        flv_pull_url = live_json.get('data').get('data')[0].get('stream_url').get('flv_pull_url')
        try:
            partition = live_json.get('data').get('partition_road_map').get('partition').get('title')
            sub_partition = live_json.get('data').get('partition_road_map').get('sub_partition').get('partition').get('title')
        except Exception as e:
            partition = '无'
            sub_partition = '无'
        print(f'[   💻   ]:直播间：{title}  当前{display_long}  主播：{nickname}  分区：{partition}--{sub_partition}  观看人数：{user_count}\r')
        flv = []
        print('[   🎦   ]:直播间清晰度')
        for (i, f) in enumerate(flv_pull_url.keys()):
            print('[   %s   ]: %s' % (i, f))
            flv.append(f)
        rate = int(input('[   🎬   ]输入数字选择推流清晰度：'))
        print('[   %s   ]:%s' % (flv[rate], flv_pull_url[flv[rate]]))
        input('[   📺   ]:复制链接使用下载工具下载，按回车退出')