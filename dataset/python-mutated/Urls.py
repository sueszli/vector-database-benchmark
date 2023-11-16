"""
@Description:Urls.py
@Date       :2023/02/08 18:14:47
@Author     :JohnserfSeed
@version    :0.0.1
@License    :MIT License
@Github     :https://github.com/johnserf-seed
@Mail       :johnserf-seed@foxmail.com
-------------------------------------------------
Change Log  :
2023/02/08 18:14:47 - Create Urls from https://johnserf-seed.github.io/DouyinApiDoc/APIdocV1.0.html
-------------------------------------------------
"""
import Util

class Urls:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.TAB_FEED = 'https://www.douyin.com/aweme/v1/web/tab/feed/?'
        self.USER_SHORT_INFO = 'https://www.douyin.com/aweme/v1/web/im/user/info/?'
        self.USER_DETAIL = 'https://www.douyin.com/aweme/v1/web/user/profile/other/?'
        self.BASE_AWEME = 'https://www.douyin.com/aweme/v1/web/aweme/'
        self.USER_POST = 'https://www.douyin.com/aweme/v1/web/aweme/post/?'
        self.POST_DETAIL = 'https://www.douyin.com/aweme/v1/web/aweme/detail/?'
        self.USER_FAVORITE_A = 'https://www.douyin.com/aweme/v1/web/aweme/favorite/?'
        self.USER_FAVORITE_B = 'https://www.iesdouyin.com/web/api/v2/aweme/like/?'
        self.USER_HISTORY = 'https://www.douyin.com/aweme/v1/web/history/read/?'
        self.USER_COLLECTION = 'https://www.douyin.com/aweme/v1/web/aweme/listcollection/?'
        self.COMMENT = 'https://www.douyin.com/aweme/v1/web/comment/list/?'
        self.FRIEND_FEED = 'https://www.douyin.com/aweme/v1/web/familiar/feed/?'
        self.FOLLOW_FEED = 'https://www.douyin.com/aweme/v1/web/follow/feed/?'
        self.RELATED = 'https://www.douyin.com/aweme/v1/web/aweme/related/?'
        self.LIVE = 'https://live.douyin.com/webcast/room/web/enter/?'
        self.SSO_LOGIN_GET_QR = 'https://sso.douyin.com/get_qrcode/?'
        self.SSO_LOGIN_CHECK_QR = 'https://sso.douyin.com/check_qrconnect/?'
        self.SSO_LOGIN_CHECK_LOGIN = 'https://sso.douyin.com/check_login/?'
        self.SSO_LOGIN_REDIRECT = 'https://www.douyin.com/login/?'
        self.SSO_LOGIN_CALLBACK = 'https://www.douyin.com/passport/sso/login/callback/?'
        self.POST_COMMENT = 'https://www.douyin.com/aweme/v1/web/comment/list/?'
        self.POST_COMMENT_PUBLISH = 'https://www.douyin.com/aweme/v1/web/comment/publish?'
        self.POST_COMMENT_DELETE = 'https://www.douyin.com/aweme/v1/web/comment/delete/?'
        self.POST_COMMENT_DIGG = 'https://www.douyin.com/aweme/v1/web/comment/digg?'
        self.POST_COMMENT_REPLY = 'https://www.douyin.com/aweme/v1/web/comment/list/reply/?'
        self.NOTICE = 'https://www.douyin.com/aweme/v1/web/notice/?'