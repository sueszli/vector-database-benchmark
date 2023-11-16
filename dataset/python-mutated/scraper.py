import random
import re
import os
import time
import execjs
import aiohttp
import httpx
import platform
import asyncio
import traceback
import configparser
import urllib.parse
import random
import json
from zlib import crc32
from typing import Union
from tenacity import *

class Scraper:
    """__________________________________________⬇️initialization(初始化)⬇️______________________________________"""

    def __init__(self):
        if False:
            print('Hello World!')
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'}
        self.douyin_api_headers = {'accept-encoding': 'gzip, deflate, br', 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36', 'Referer': 'https://www.douyin.com/', 'cookie': 's_v_web_id=verify_ln4g95yq_8yd5gq1d_ZOJz_4i0Z_8g5H_VnqOInAXfDjQ; ttwid=1%7CLOO5jA3xKFP2HUC4tFAnPpFGRifnKCdQ8kuwwY24h9Y%7C1695982617%7C032f9efe9aef7c1a3ec2fd13f460a3565f556fd68c6b227985c65747e3111a28; passport_csrf_token=476087cea19a0f2fef97fd384e922f80; passport_csrf_token_default=476087cea19a0f2fef97fd384e922f80; __ac_nonce=06529d73200a6acdd9289; __ac_signature=_02B4Z6wo00f01JsrSRgAAIDB2BvQeRHxXJSbG02AAEPnsyJBRv.Ek0Zo3rTJH9wE4R9g7KoeoPrwt65OLUSkHPTL-BDM5rxgepgijGI7BNe5hZ7zNiKIUK556QwDMuFLZa-fn2SNAlctY4Cxee; douyin.com; device_web_cpu_core=16; device_web_memory_size=-1; architecture=amd64; webcast_local_quality=null; IsDouyinActive=true; home_can_add_dy_2_desktop=%220%22; strategyABtestKey=%221697240884.429%22; stream_recommend_feed_params=%22%7B%5C%22cookie_enabled%5C%22%3Atrue%2C%5C%22screen_width%5C%22%3A1344%2C%5C%22screen_height%5C%22%3A756%2C%5C%22browser_online%5C%22%3Atrue%2C%5C%22cpu_core_num%5C%22%3A16%2C%5C%22device_memory%5C%22%3A0%2C%5C%22downlink%5C%22%3A%5C%22%5C%22%2C%5C%22effective_type%5C%22%3A%5C%22%5C%22%2C%5C%22round_trip_time%5C%22%3A0%7D%22; VIDEO_FILTER_MEMO_SELECT=%7B%22expireTime%22%3A1697845684695%2C%22type%22%3A1%7D; volume_info=%7B%22isUserMute%22%3Afalse%2C%22isMute%22%3Atrue%2C%22volume%22%3A0.5%7D; FORCE_LOGIN=%7B%22videoConsumedRemainSeconds%22%3A180%7D; csrf_session_id=6f34e666e71445c9d39d8d06a347a13f; bd_ticket_guard_client_data=eyJiZC10aWNrZXQtZ3VhcmQtdmVyc2lvbiI6MiwiYmQtdGlja2V0LWd1YXJkLWl0ZXJhdGlvbi12ZXJzaW9uIjoxLCJiZC10aWNrZXQtZ3VhcmQtcmVlLXB1YmxpYy1rZXkiOiJCTFFUdWdBbEg4Q1NxRENRdE9QdnN6K1pSOVBjdnBCOWg5dlp1VDhSRU1qSFFVNEVia2dOYnRHR0pBZFZ3c1hiak5EV01WTjBXd05CWEtSbTBWNDI4eHc9IiwiYmQtdGlja2V0LWd1YXJkLXdlYi12ZXJzaW9uIjoxfQ%3D%3D; msToken=O0WY2EiVqldmSETtrN2lLnKHeFHvy5xyKf0_Wj7xHUTTb6eMsV47NNy8TAvCw-BzjJu3EHLYLQ_F57RJI9TIIGxpl72LOqU3JKD2mSCNRK7bRdpj5OCMelAW7zA=; msToken=B1N9FM825TkvFbayDsDvZxM8r5suLrsfQbC93TciS0O9Iii8iJpAPd__FM2rpLUJi5xtMencSXLeNn8xmOS9q7bP0CUsrt9oVTL08YXLPRzZm0dHKLc9PGRlyEk=; tt_scid=CB3bLQLXQ7-hdquJoiVfLG426BLihcDygWOyFenygGFyeyJ3doSH1iYdwaR3kq0Ta886'}
        self.tiktok_api_headers = {'User-Agent': 'com.ss.android.ugc.trill/494+Mozilla/5.0+(Linux;+Android+12;+2112123G+Build/SKQ1.211006.001;+wv)+AppleWebKit/537.36+(KHTML,+like+Gecko)+Version/4.0+Chrome/107.0.5304.105+Mobile+Safari/537.36'}
        self.bilibili_api_headers = {'User-Agent': 'com.ss.android.ugc.trill/494+Mozilla/5.0+(Linux;+Android+12;+2112123G+Build/SKQ1.211006.001;+wv)+AppleWebKit/537.36+(KHTML,+like+Gecko)+Version/4.0+Chrome/107.0.5304.105+Mobile+Safari/537.36'}
        self.ixigua_api_headers = {'authority': 'ib.365yg.com', 'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7', 'accept-language': 'zh-CN,zh;q=0.9', 'cache-control': 'no-cache', 'pragma': 'no-cache', 'sec-ch-ua': '"Chromium";v="116", "Not)A;Brand";v="24", "Google Chrome";v="116"', 'sec-ch-ua-mobile': '?0', 'sec-ch-ua-platform': '"macOS"', 'sec-fetch-dest': 'document', 'sec-fetch-mode': 'navigate', 'sec-fetch-site': 'none', 'sec-fetch-user': '?1', 'upgrade-insecure-requests': '1', 'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'}
        self.kuaishou_api_headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7', 'Accept-Language': 'zh-CN,zh;q=0.9', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive', 'Cookie': 'kpf=PC_WEB; clientid=3; did=web_c5627223fe1e796669894e6cb74f1461; _ga=GA1.1.1139357938.1696318390; didv=1696329758000; _ga_0P9YPW1GQ3=GS1.1.1696659232.14.0.1696659232.0.0.0; kpn=KUAISHOU_VISION', 'Pragma': 'no-cache', 'Sec-Fetch-Dest': 'document', 'Sec-Fetch-Mode': 'navigate', 'Sec-Fetch-Site': 'none', 'Sec-Fetch-User': '?1', 'Upgrade-Insecure-Requests': '1', 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36', 'sec-ch-ua': '"Google Chrome";v="117", "Not;A=Brand";v="8", "Chromium";v="117"', 'sec-ch-ua-mobile': '?0', 'sec-ch-ua-platform': '"macOS"'}
        if os.path.exists('config.ini'):
            self.config = configparser.ConfigParser()
            self.config.read('config.ini', encoding='utf-8')
            if self.config['Scraper']['Proxy_switch'] == 'True':
                if self.config['Scraper']['Use_different_protocols'] == 'False':
                    self.proxies = {'all': self.config['Scraper']['All']}
                else:
                    self.proxies = {'http': self.config['Scraper']['Http_proxy'], 'https': self.config['Scraper']['Https_proxy']}
            else:
                self.proxies = None
        else:
            self.proxies = None
        if platform.system() == 'Windows':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    '__________________________________________⬇️utils(实用程序)⬇️______________________________________'

    @staticmethod
    def get_url(text: str) -> Union[str, None]:
        if False:
            print('Hello World!')
        try:
            url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
            if len(url) > 0:
                return url[0]
        except Exception as e:
            print('Error in get_url:', e)
            return None

    @staticmethod
    def relpath(file):
        if False:
            while True:
                i = 10
        ' Always locate to the correct relative path. '
        from sys import _getframe
        from pathlib import Path
        frame = _getframe(1)
        curr_file = Path(frame.f_code.co_filename)
        return str(curr_file.parent.joinpath(file).resolve())

    @staticmethod
    def generate_x_bogus_url(url: str, headers: dict) -> str:
        if False:
            print('Hello World!')
        query = urllib.parse.urlparse(url).query
        xbogus = execjs.compile(open('./X-Bogus.js').read()).call('sign', query, headers['User-Agent'])
        new_url = url + '&X-Bogus=' + xbogus
        return new_url

    @retry(stop=stop_after_attempt(4), wait=wait_fixed(7))
    async def convert_share_urls(self, url: str) -> Union[str, None]:
        """
        用于将分享链接(短链接)转换为原始链接/Convert share links (short links) to original links
        :return: 原始链接/Original link
        """
        url = self.get_url(url)
        if url is None:
            print('无法检索到链接/Unable to retrieve link')
            return None
        if 'douyin' in url:
            '\n            抖音视频链接类型(不全)：\n            1. https://v.douyin.com/MuKhKn3/\n            2. https://www.douyin.com/video/7157519152863890719\n            3. https://www.iesdouyin.com/share/video/7157519152863890719/?region=CN&mid=7157519152863890719&u_code=ffe6jgjg&titleType=title&timestamp=1600000000&utm_source=copy_link&utm_campaign=client_share&utm_medium=android&app=aweme&iid=123456789&share_id=123456789\n            抖音用户链接类型(不全)：\n            1. https://www.douyin.com/user/MS4wLjABAAAAbLMPpOhVk441et7z7ECGcmGrK42KtoWOuR0_7pLZCcyFheA9__asY-kGfNAtYqXR?relation=0&vid=7157519152863890719\n            2. https://v.douyin.com/MuKoFP4/\n            抖音直播链接类型(不全)：\n            1. https://live.douyin.com/88815422890\n            '
            if 'v.douyin' in url:
                url = re.compile('(https://v.douyin.com/)\\w+', re.I).match(url).group()
                print('正在通过抖音分享链接获取原始链接...')
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, headers=self.headers, proxy=self.proxies, allow_redirects=False, timeout=10) as response:
                            if response.status == 302:
                                url = response.headers['Location'].split('?')[0] if '?' in response.headers['Location'] else response.headers['Location']
                                print('获取原始链接成功, 原始链接为: {}'.format(url))
                                return url
                except Exception as e:
                    print('获取原始链接失败！')
                    print(e)
                    raise e
            else:
                print('该链接为原始链接,无需转换,原始链接为: {}'.format(url))
                return url
        elif 'tiktok' in url:
            '\n            TikTok视频链接类型(不全)：\n            1. https://www.tiktok.com/@tiktok/video/6950000000000000000\n            2. https://www.tiktok.com/t/ZTRHcXS2C/\n            TikTok用户链接类型(不全)：\n            1. https://www.tiktok.com/@tiktok\n            '
            if '@' in url:
                print('该链接为原始链接,无需转换,原始链接为: {}'.format(url))
                return url
            else:
                print('正在通过TikTok分享链接获取原始链接...')
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, headers=self.headers, proxy=self.proxies, allow_redirects=False, timeout=10) as response:
                            if response.status == 301:
                                url = response.headers['Location'].split('?')[0] if '?' in response.headers['Location'] else response.headers['Location']
                                print('获取原始链接成功, 原始链接为: {}'.format(url))
                                return url
                except Exception as e:
                    print('获取原始链接失败！')
                    print(e)
                    return None
        elif 'b23.tv' in url or 'bilibili' in url:
            '\n            bilibili视频链接类型(不全)：\n            1. https://b23.tv/Ya65brl\n            2. https://www.bilibili.com/video/BV1MK4y1w7MV/\n            bilibili用户链接类型(不全)：\n            1. https://www.douyin.com/user/MS4wLjABAAAAbLMPpOhVk441et7z7ECGcmGrK42KtoWOuR0_7pLZCcyFheA9__asY-kGfNAtYqXR?relation=0&vid=7157519152863890719\n            bilibili直播链接类型(不全)：\n            '
            if 'b23.tv' in url:
                print('正在通过哔哩哔哩分享链接获取原始链接...')
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, headers=self.headers, proxy=self.proxies, allow_redirects=False, timeout=10) as response:
                            if response.status == 302:
                                url = response.headers['Location'].split('?')[0] if '?' in response.headers['Location'] else response.headers['Location']
                                print('获取原始链接成功, 原始链接为: {}'.format(url))
                                return url
                except Exception as e:
                    print('获取原始链接失败！')
                    print(e)
                    raise e
            else:
                print('该链接为原始链接,无需转换,原始链接为: {}'.format(url))
                return url
        elif 'ixigua.com' in url:
            '\n            西瓜视频链接类型(不全)：\n            1. https://v.ixigua.com/ienrQ5bR/\n            2. https://www.ixigua.com/7270448082586698281\n            3. https://m.ixigua.com/video/7270448082586698281\n            西瓜用户链接类型(不全)：\n            1. https://www.ixigua.com/home/3189050062678823\n            西瓜直播链接类型(不全)：\n            '
            if 'v.ixigua.com' in url:
                print('正在通过西瓜分享链接获取原始链接...')
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, headers=self.ixigua_api_headers, proxy=self.proxies, allow_redirects=False, timeout=10) as response:
                            if response.status == 302:
                                url = response.headers['Location'].split('?')[0] if '?' in response.headers['Location'] else response.headers['Location']
                                print('获取原始链接成功, 原始链接为: {}'.format(url))
                                return url
                except Exception as e:
                    print('获取原始链接失败！')
                    print(e)
                    raise e
            else:
                print('该链接为原始链接,无需转换,原始链接为: {}'.format(url))
                return url
        elif 'kuaishou.com' in url:
            '\n            快手视频链接类型(不全)：\n            1. https://www.kuaishou.com/short-video/3xiqjrezhqjyzxw\n            2. https://v.kuaishou.com/75kDOJ \n            快手用户链接类型(不全)：\n            1. https://www.kuaishou.com/profile/3xvgbyksme9f2p6\n            快手直播链接类型(不全)：\n            1.https://live.kuaishou.com/u/3xv5uz3ui6iga5w\n            2.https://v.kuaishou.com/5Ch22V\n            '
            if 'v.kuaishou.com' in url:
                print('正在通过快手分享链接获取原始链接...')
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, headers=self.kuaishou_api_headers, proxy=self.proxies, allow_redirects=False, timeout=10) as response:
                            if response.status == 302:
                                url = response.headers['Location'].split('?')[0] if '?' in response.headers['Location'] else response.headers['Location']
                                print('获取原始链接成功, 原始链接为: {}'.format(url))
                                return url
                except Exception as e:
                    print('获取原始链接失败！')
                    print(e)
                    raise e
            else:
                print('该链接为原始链接,无需转换,原始链接为: {}'.format(url))
                return url
    '__________________________________________⬇️Douyin methods(抖音方法)⬇️______________________________________'

    def generate_x_bogus_url(self, url: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        生成抖音X-Bogus签名\n        :param url: 视频链接\n        :return: 包含X-Bogus签名的URL\n        '
        query = urllib.parse.urlparse(url).query
        xbogus = execjs.compile(open(self.relpath('./X-Bogus.js')).read()).call('sign', query, self.headers['User-Agent'])
        new_url = url + '&X-Bogus=' + xbogus
        return new_url

    async def get_douyin_video_id(self, original_url: str) -> Union[str, None]:
        """
        获取视频id
        :param original_url: 视频链接
        :return: 视频id
        """
        try:
            video_url = await self.convert_share_urls(original_url)
            if '/video/' in video_url:
                key = re.findall('/video/(\\d+)?', video_url)[0]
                return key
            elif 'discover?' in video_url:
                key = re.findall('modal_id=(\\d+)', video_url)[0]
                return key
            elif 'live.douyin' in video_url:
                video_url = video_url.split('?')[0] if '?' in video_url else video_url
                key = video_url.replace('https://live.douyin.com/', '')
                return key
            elif 'note' in video_url:
                key = re.findall('/note/(\\d+)?', video_url)[0]
                return key
        except Exception as e:
            print('获取抖音视频ID出错了:{}'.format(e))
            return None

    @retry(stop=stop_after_attempt(4), wait=wait_fixed(7))
    async def get_douyin_video_data(self, video_id: str) -> Union[dict, None]:
        """
        :param video_id: str - 抖音视频id
        :return:dict - 包含信息的字典
        """
        try:
            api_url = f'https://www.douyin.com/aweme/v1/web/aweme/detail/?device_platform=webapp&aid=6383&channel=channel_pc_web&aweme_id={video_id}&pc_client_type=1&version_code=190500&version_name=19.5.0&cookie_enabled=true&screen_width=1344&screen_height=756&browser_language=zh-CN&browser_platform=Win32&browser_name=Firefox&browser_version=118.0&browser_online=true&engine_name=Gecko&engine_version=109.0&os_name=Windows&os_version=10&cpu_core_num=16&device_memory=&platform=PC&webid=7284189800734082615&msToken=B1N9FM825TkvFbayDsDvZxM8r5suLrsfQbC93TciS0O9Iii8iJpAPd__FM2rpLUJi5xtMencSXLeNn8xmOS9q7bP0CUsrt9oVTL08YXLPRzZm0dHKLc9PGRlyEk='
            api_url = self.generate_x_bogus_url(api_url)
            print('正在请求抖音视频API: {}'.format(api_url))
            async with aiohttp.ClientSession() as session:
                self.douyin_api_headers['Referer'] = f'https://www.douyin.com/video/{video_id}'
                async with session.get(api_url, headers=self.douyin_api_headers, proxy=self.proxies, timeout=10) as response:
                    response = await response.json()
                    video_data = response['aweme_detail']
                    return video_data
        except Exception as e:
            raise ValueError(f'获取抖音视频数据出错了: {e}')

    @retry(stop=stop_after_attempt(4), wait=wait_fixed(7))
    async def get_douyin_live_video_data(self, web_rid: str) -> Union[dict, None]:
        try:
            api_url = f'https://live.douyin.com/webcast/web/enter/?aid=6383&web_rid={web_rid}'
            print('正在请求抖音直播API: {}'.format(api_url))
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, headers=self.douyin_api_headers, proxy=self.proxies, timeout=10) as response:
                    response = await response.json()
                    video_data = response['data']
                    print(video_data)
                    print('获取视频数据成功！')
                    return video_data
        except Exception as e:
            print('获取抖音视频数据失败！原因:{}'.format(e))
            raise e
    '__________________________________________⬇️TikTok methods(TikTok方法)⬇️______________________________________'

    async def get_tiktok_video_id(self, original_url: str) -> Union[str, None]:
        """
        获取视频id
        :param original_url: 视频链接
        :return: 视频id
        """
        try:
            original_url = await self.convert_share_urls(original_url)
            if '/video/' in original_url:
                video_id = re.findall('/video/(\\d+)', original_url)[0]
            elif '/v/' in original_url:
                video_id = re.findall('/v/(\\d+)', original_url)[0]
            return video_id
        except Exception as e:
            print('获取TikTok视频ID出错了:{}'.format(e))
            return None

    @retry(stop=stop_after_attempt(4), wait=wait_fixed(7))
    async def get_tiktok_video_data(self, video_id: str) -> Union[dict, None]:
        """
        获取单个视频信息
        :param video_id: 视频id
        :return: 视频信息
        """
        try:
            api_url = f'https://api16-normal-c-useast1a.tiktokv.com/aweme/v1/feed/?aweme_id={video_id}'
            print('正在获取视频数据API: {}'.format(api_url))
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, headers=self.tiktok_api_headers, proxy=self.proxies, timeout=10) as response:
                    response = await response.json()
                    video_data = response['aweme_list'][0]
                    return video_data
        except Exception as e:
            print('获取视频信息失败！原因:{}'.format(e))
            raise e
    '__________________________________________⬇️bilibili methods(Bilibili方法)⬇️______________________________________'

    async def get_bilibili_video_id(self, original_url: str) -> Union[str, None]:
        """
        获取视频id
        :param original_url: 视频链接
        :return: 视频id
        """
        try:
            original_url = await self.convert_share_urls(original_url)
            if 'video/BV' in original_url:
                match = re.search('video/BV(?P<id>[0-9a-zA-Z]+)', original_url)
                if match:
                    return f"video/BV{match.group('id')}"
            elif 'video/av' in original_url:
                match = re.search('video/av(?P<id>[0-9a-zA-Z]+)', original_url)
                if match:
                    return f"video/av{match.group('id')}"
            return None
        except Exception as e:
            raise ValueError(f'获取BiliBili视频ID出错了:{e}')

    @retry(stop=stop_after_attempt(4), wait=wait_fixed(7))
    async def get_bilibili_video_data(self, video_id: str) -> Union[dict, None]:
        """
        获取单个视频信息
        :param video_id: 视频id
        :return: 视频信息
        """
        print('正在获取BiliBili视频数据...')
        try:
            api_url = f"https://api.bilibili.com/x/web-interface/view?bvid={video_id.replace('video/BV', '')}"
            if 'video/av' in video_id:
                api_url = f"https://api.bilibili.com/x/web-interface/view?aid={video_id.replace('video/av', '')}"
            print(f'正在获取视频数据API: {api_url}')
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, headers=self.bilibili_api_headers, proxy=self.proxies, timeout=10) as response:
                    response = await response.json()
                    avid = response.get('data', {}).get('aid', '')
                    cid = response.get('data', {}).get('cid', '')
                    print('获取视频信息成功！')
            play_url_api = f'https://api.bilibili.com/x/player/playurl?avid={avid}&cid={cid}&platform=html5'
            async with aiohttp.ClientSession() as session:
                async with session.get(play_url_api, headers=self.bilibili_api_headers, proxy=self.proxies, timeout=10) as response:
                    response = await response.json()
                    video_data = response.get('data', {}).get('durl', [])[0]['url']
                    video_data = {'status': 'success', 'message': '更多接口请查看(More API see): https://api.tikhub.io/', 'type': 'video', 'platform': 'bilibili', 'video_url': video_data}
            return video_data
        except Exception as e:
            raise ValueError(f'获取BiliBili视频数据出错了:{e}')
    '__________________________________________⬇️xigua methods(xigua方法)⬇️______________________________________'

    def get_xigua_json_url(self, video_id):
        if False:
            for i in range(10):
                print('nop')
        r = str(random.random())[2:]
        url_part = '/video/urls/v/1/toutiao/mp4/{}?r={}'.format(video_id, r)
        s = crc32(url_part.encode())
        json_url = 'https://ib.365yg.com{}&s={}&nobase64=true'.format(url_part, s)
        return json_url

    async def get_ixigua_video_id(self, original_url: str) -> Union[str, None]:
        """
        获取视频id
        :param original_url: 视频链接
        :return: 视频id
        """
        try:
            original_url = await self.convert_share_urls(original_url)
            if 'www.ixigua.com/' in original_url:
                video_id = re.findall('ixigua\\.com/(\\d+)', original_url)[0]
            elif 'm.ixigua.com/video' in original_url:
                video_id = re.findall('/video/(\\d+)', original_url)[0]
            return video_id
        except Exception as e:
            raise ValueError(f'获取西瓜视频ID出错了:{e}')

    @retry(stop=stop_after_attempt(4), wait=wait_fixed(7))
    async def get_ixigua_video_data(self, video_id: str) -> Union[dict, None]:
        """
        获取单个视频信息
        :param video_id: 视频id
        :return: 视频信息
        """
        print('正在获取西瓜视频数据...')
        try:
            video_url = f'https://m.ixigua.com/video/{video_id}?wid_try=1'
            print('video_url', video_url)
            async with aiohttp.ClientSession() as session:
                async with session.get(video_url, headers=self.ixigua_api_headers, proxy=self.proxies, timeout=10) as response:
                    response = await response.text()
                    search = re.search('"vid":"([^"]+)",', response)
                    vid = search.group(1)
                    print('获取视频vid信息成功！')
                    play_url_api = self.get_xigua_json_url(vid)
            print(f'正在获取视频数据API: {play_url_api}')
            async with aiohttp.ClientSession() as session:
                async with session.get(play_url_api, headers=self.ixigua_api_headers, proxy=self.proxies, timeout=10) as response:
                    response = await response.json()
                    video_data = response.get('data', {}).get('video_list', {}).get('video_3', {}).get('main_url', '')
                    video_data = {'status': 'success', 'message': '更多接口请查看(More API see): https://api.tikhub.io/', 'type': 'video', 'platform': '西瓜', 'video_url': video_data}
            return video_data
        except Exception as e:
            raise ValueError(f'获取西瓜视频数据出错了:{e}')
    '__________________________________________⬇️kuaishou methods(kuaishou方法)⬇️______________________________________'

    async def get_kuaishou_video_id(self, original_url: str) -> Union[str, None]:
        """
        获取视频id
        :param original_url: 视频链接
        :return: 视频id
        """
        try:
            original_url = await self.convert_share_urls(original_url)
            if '/fw/photo/' in original_url:
                video_id = re.findall('/fw/photo/(.*)', original_url)[0]
            elif 'short-video' in original_url:
                video_id = re.findall('short-video/(.*)', original_url)[0]
            return video_id
        except Exception as e:
            raise ValueError(f'获取快手视频ID出错了:{e}')

    @retry(stop=stop_after_attempt(4), wait=wait_fixed(7))
    async def get_kuaishou_video_data(self, video_id: str) -> Union[dict, None]:
        """
        获取单个视频信息
        :param video_id: 视频id
        :return: 视频信息
        """
        print('正在获取快手视频数据...')
        try:
            video_url = f'https://www.kuaishou.com/short-video/{video_id}'
            print('video_url', video_url)
            print(f'正在获取视频数据API: {video_url}')
            async with aiohttp.ClientSession() as session:
                async with session.get(video_url, headers=self.kuaishou_api_headers, proxy=self.proxies, timeout=10) as response:
                    response = await response.text()
                    video_data = re.findall('"photoH265Url":"(.*?)"', response)[0]
                    if video_data:
                        video_data = video_data.encode().decode('raw_unicode-escape')
                    video_data = {'status': 'success', 'message': '更多接口请查看(More API see): https://api.tikhub.io/', 'type': 'video', 'platform': '快手', 'video_url': video_data}
            return video_data
        except Exception as e:
            raise ValueError(f'获取快手视频数据出错了:{e}')
    '__________________________________________⬇️Hybrid methods(混合方法)⬇️______________________________________'

    async def judge_url_platform(self, video_url: str) -> str:
        if 'douyin' in video_url:
            url_platform = 'douyin'
        elif 'bilibili' in video_url or 'b23.tv' in video_url:
            url_platform = 'bilibili'
        elif 'xigua' in video_url:
            url_platform = 'xigua'
        elif 'kuaishou' in video_url:
            url_platform = 'kuaishou'
        elif 'tiktok' in video_url:
            url_platform = 'tiktok'
        else:
            url_platform = None
        return url_platform

    async def hybrid_parsing(self, video_url: str) -> dict:
        url_platform = await self.judge_url_platform(video_url)
        if not url_platform:
            raise ValueError(f'链接**{video_url}**不是抖音、Bilibili、西瓜、快手、TikTok链接！')
        print(f'正在解析**{url_platform}**视频链接...')
        video_id = await self.get_douyin_video_id(video_url) if url_platform == 'douyin' else await self.get_tiktok_video_id(video_url) if url_platform == 'tiktok' else await self.get_bilibili_video_id(video_url) if url_platform == 'bilibili' else await self.get_ixigua_video_id(video_url) if url_platform == 'xigua' else await self.get_kuaishou_video_id(video_url) if url_platform == 'kuaishou' else None
        if not video_id:
            raise ValueError(f'获取**{url_platform}**视频ID失败！')
        print(f'获取到的**{url_platform}**视频ID是{video_id}')
        data = await self.get_douyin_video_data(video_id) if url_platform == 'douyin' else await self.get_tiktok_video_data(video_id) if url_platform == 'tiktok' else await self.get_bilibili_video_data(video_id) if url_platform == 'bilibili' else await self.get_ixigua_video_data(video_id) if url_platform == 'xigua' else await self.get_kuaishou_video_data(video_id) if url_platform == 'kuaishou' else None
        if data:
            if url_platform == 'bilibili':
                print('获取Bilibili视频数据成功！')
                return data
            if url_platform == 'xigua':
                print('获取西瓜视频数据成功！')
                return data
            if url_platform == 'kuaishou':
                print('获取快手视频数据成功！')
                return data
            print(f'获取**{url_platform}**视频数据成功，正在判断数据类型...')
            url_type_code = data['aweme_type']
            url_type_code_dict = {2: 'image', 4: 'video', 68: 'image', 0: 'video', 51: 'video', 55: 'video', 58: 'video', 61: 'video', 150: 'image'}
            url_type = url_type_code_dict.get(url_type_code, 'video')
            print(f'获取到的**{url_platform}**的链接类型是{url_type}')
            print('准备开始判断并处理数据...')
            '\n            以下为(视频||图片)数据处理的四个方法,如果你需要自定义数据处理请在这里修改.\n            The following are four methods of (video || image) data processing. \n            If you need to customize data processing, please modify it here.\n            '
            '\n            创建已知数据字典(索引相同)，稍后使用.update()方法更新数据\n            Create a known data dictionary (index the same), \n            and then use the .update() method to update the data\n            '
            result_data = {'status': 'success', 'message': '更多接口请查看(More API see): https://api.tikhub.io/', 'type': url_type, 'platform': url_platform, 'aweme_id': video_id, 'official_api_url': {'User-Agent': self.headers['User-Agent'], 'api_url': f'https://www.iesdouyin.com/aweme/v1/web/aweme/detail/?aweme_id={video_id}&aid=1128&version_name=23.5.0&device_platform=android&os_version=2333&Github=Evil0ctal&words=FXXK_U_ByteDance'} if url_platform == 'douyin' else {'User-Agent': self.tiktok_api_headers['User-Agent'], 'api_url': f'https://api16-normal-c-useast1a.tiktokv.com/aweme/v1/feed/?aweme_id={video_id}'}, 'desc': data.get('desc'), 'create_time': data.get('create_time'), 'author': data.get('author'), 'music': data.get('music'), 'statistics': data.get('statistics'), 'cover_data': {'cover': data.get('video').get('cover'), 'origin_cover': data.get('video').get('origin_cover'), 'dynamic_cover': data.get('video').get('dynamic_cover')}, 'hashtags': data.get('text_extra')}
            api_data = None
            try:
                if url_platform == 'douyin':
                    if url_type == 'video':
                        print('正在处理抖音视频数据...')
                        uri = data['video']['play_addr']['uri']
                        wm_video_url = data['video']['play_addr']['url_list'][0]
                        wm_video_url_HQ = f'https://aweme.snssdk.com/aweme/v1/playwm/?video_id={uri}&radio=1080p&line=0'
                        nwm_video_url = wm_video_url.replace('playwm', 'play')
                        nwm_video_url_HQ = f'https://aweme.snssdk.com/aweme/v1/play/?video_id={uri}&ratio=1080p&line=0'
                        api_data = {'video_data': {'wm_video_url': wm_video_url, 'wm_video_url_HQ': wm_video_url_HQ, 'nwm_video_url': nwm_video_url, 'nwm_video_url_HQ': nwm_video_url_HQ}}
                    elif url_type == 'image':
                        print('正在处理抖音图片数据...')
                        no_watermark_image_list = []
                        watermark_image_list = []
                        for i in data['images']:
                            no_watermark_image_list.append(i['url_list'][0])
                            watermark_image_list.append(i['download_url_list'][0])
                        api_data = {'image_data': {'no_watermark_image_list': no_watermark_image_list, 'watermark_image_list': watermark_image_list}}
                elif url_platform == 'tiktok':
                    if url_type == 'video':
                        print('正在处理TikTok视频数据...')
                        wm_video = data['video']['download_addr']['url_list'][0]
                        api_data = {'video_data': {'wm_video_url': wm_video, 'wm_video_url_HQ': wm_video, 'nwm_video_url': data['video']['play_addr']['url_list'][0], 'nwm_video_url_HQ': data['video']['bit_rate'][0]['play_addr']['url_list'][0]}}
                    elif url_type == 'image':
                        print('正在处理TikTok图片数据...')
                        no_watermark_image_list = []
                        watermark_image_list = []
                        for i in data['image_post_info']['images']:
                            no_watermark_image_list.append(i['display_image']['url_list'][0])
                            watermark_image_list.append(i['owner_watermark_image']['url_list'][0])
                        api_data = {'image_data': {'no_watermark_image_list': no_watermark_image_list, 'watermark_image_list': watermark_image_list}}
                result_data.update(api_data)
                return result_data
            except Exception as e:
                traceback.print_exc()
                print('数据处理失败！')
                return {'status': 'failed', 'message': '数据处理失败！/Data processing failed!'}
        else:
            print('[抖音|TikTok方法]返回数据为空，无法处理！')
            return {'status': 'failed', 'message': '返回数据为空，无法处理！/Return data is empty and cannot be processed!'}

    @staticmethod
    def hybrid_parsing_minimal(data: dict) -> dict:
        if False:
            i = 10
            return i + 15
        if data['status'] == 'success':
            result = {'status': 'success', 'message': data.get('message'), 'platform': data.get('platform'), 'type': data.get('type'), 'desc': data.get('desc'), 'wm_video_url': data['video_data']['wm_video_url'] if data['type'] == 'video' else None, 'wm_video_url_HQ': data['video_data']['wm_video_url_HQ'] if data['type'] == 'video' else None, 'nwm_video_url': data['video_data']['nwm_video_url'] if data['type'] == 'video' else None, 'nwm_video_url_HQ': data['video_data']['nwm_video_url_HQ'] if data['type'] == 'video' else None, 'no_watermark_image_list': data['image_data']['no_watermark_image_list'] if data['type'] == 'image' else None, 'watermark_image_list': data['image_data']['watermark_image_list'] if data['type'] == 'image' else None}
            return result
        else:
            return data
'__________________________________________⬇️Test methods(测试方法)⬇️______________________________________'

async def async_test(_douyin_url: str=None, _tiktok_url: str=None, _bilibili_url: str=None, _ixigua_url: str=None, _kuaishou_url: str=None) -> None:
    start_time = time.time()
    print('<异步测试/Async test>')
    print('\n--------------------------------------------------')
    print('正在测试异步获取快手视频ID方法...')
    kuaishou_id = await api.get_kuaishou_video_id(_kuaishou_url)
    print(f'快手视频ID: {kuaishou_id}')
    print('正在测试异步获取快手视频数据方法...')
    kuaishou_data = await api.get_kuaishou_video_data(kuaishou_id)
    print(f'快手视频数据: {str(kuaishou_data)}')
    print('\n--------------------------------------------------')
    print('正在测试异步获取西瓜视频ID方法...')
    ixigua_id = await api.get_ixigua_video_id(_ixigua_url)
    print(f'西瓜视频ID: {ixigua_id}')
    print('正在测试异步获取西瓜视频数据方法...')
    ixigua_data = await api.get_ixigua_video_data(ixigua_id)
    print(f'西瓜视频数据: {str(ixigua_data)[:100]}')
    print('\n--------------------------------------------------')
    print('正在测试异步获取哔哩哔哩视频ID方法...')
    bilibili_id = await api.get_bilibili_video_id(_bilibili_url)
    print(f'哔哩哔哩视频ID: {bilibili_id}')
    print('正在测试异步获取哔哩哔哩视频数据方法...')
    bilibili_data = await api.get_bilibili_video_data(bilibili_id)
    print(f'哔哩哔哩视频数据: {str(bilibili_data)[:100]}')
    print('\n--------------------------------------------------')
    print('正在测试异步获取抖音视频ID方法...')
    douyin_id = await api.get_douyin_video_id(_douyin_url)
    print(f'抖音视频ID: {douyin_id}')
    print('正在测试异步获取抖音视频数据方法...')
    douyin_data = await api.get_douyin_video_data(douyin_id)
    print(f'抖音视频数据: {str(douyin_data)[:100]}')
    print('\n--------------------------------------------------')
    print('正在测试异步获取TikTok视频ID方法...')
    tiktok_id = await api.get_tiktok_video_id(_tiktok_url)
    print(f'TikTok视频ID: {tiktok_id}')
    print('正在测试异步获取TikTok视频数据方法...')
    tiktok_data = await api.get_tiktok_video_data(tiktok_id)
    print(f'TikTok视频数据: {str(tiktok_data)[:100]}')
    print('\n--------------------------------------------------')
    print('正在测试异步混合解析方法...')
    douyin_hybrid_data = await api.hybrid_parsing(_douyin_url)
    tiktok_hybrid_data = await api.hybrid_parsing(_tiktok_url)
    bilibili_hybrid_data = await api.hybrid_parsing(_bilibili_url)
    xigua_hybrid_data = await api.hybrid_parsing(_ixigua_url)
    kuaishou_hybrid_data = await api.hybrid_parsing(_kuaishou_url)
    print(f'抖音、TikTok、哔哩哔哩、西瓜、快手快手混合解析全部成功！')
    print('\n--------------------------------------------------')
    total_time = round(time.time() - start_time, 2)
    print('异步测试完成，总耗时: {}s'.format(total_time))
if __name__ == '__main__':
    api = Scraper()
    douyin_url = 'https://v.douyin.com/rLyrQxA/6.66'
    tiktok_url = 'https://www.tiktok.com/@evil0ctal/video/7217027383390555438'
    bilibili_url = 'https://www.bilibili.com/video/BV1Th411x7ii/'
    ixigua_url = 'https://www.ixigua.com/7270448082586698281'
    kuaishou_url = 'https://www.kuaishou.com/short-video/3xiqjrezhqjyzxw'
    asyncio.run(api.get_douyin_video_data('https://v.douyin.com/rLyrQxA/'))