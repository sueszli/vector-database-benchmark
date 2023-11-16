import logging
import os
import re
import jsonpickle
from django.conf import settings
from werobot import WeRoBot
from werobot.replies import ArticlesReply, Article
from werobot.session.filestorage import FileStorage
from djangoblog.utils import get_sha256
from servermanager.api.blogapi import BlogApi
from servermanager.api.commonapi import ChatGPT, CommandHandler
from .MemcacheStorage import MemcacheStorage
robot = WeRoBot(token=os.environ.get('DJANGO_WEROBOT_TOKEN') or 'lylinux', enable_session=True)
memstorage = MemcacheStorage()
if memstorage.is_available:
    robot.config['SESSION_STORAGE'] = memstorage
else:
    if os.path.exists(os.path.join(settings.BASE_DIR, 'werobot_session')):
        os.remove(os.path.join(settings.BASE_DIR, 'werobot_session'))
    robot.config['SESSION_STORAGE'] = FileStorage(filename='werobot_session')
blogapi = BlogApi()
cmd_handler = CommandHandler()
logger = logging.getLogger(__name__)

def convert_to_article_reply(articles, message):
    if False:
        return 10
    reply = ArticlesReply(message=message)
    from blog.templatetags.blog_tags import truncatechars_content
    for post in articles:
        imgs = re.findall('(?:http\\:|https\\:)?\\/\\/.*\\.(?:png|jpg)', post.body)
        imgurl = ''
        if imgs:
            imgurl = imgs[0]
        article = Article(title=post.title, description=truncatechars_content(post.body), img=imgurl, url=post.get_full_url())
        reply.add_article(article)
    return reply

@robot.filter(re.compile('^\\?.*'))
def search(message, session):
    if False:
        for i in range(10):
            print('nop')
    s = message.content
    searchstr = str(s).replace('?', '')
    result = blogapi.search_articles(searchstr)
    if result:
        articles = list(map(lambda x: x.object, result))
        reply = convert_to_article_reply(articles, message)
        return reply
    else:
        return '没有找到相关文章。'

@robot.filter(re.compile('^category\\s*$', re.I))
def category(message, session):
    if False:
        for i in range(10):
            print('nop')
    categorys = blogapi.get_category_lists()
    content = ','.join(map(lambda x: x.name, categorys))
    return '所有文章分类目录：' + content

@robot.filter(re.compile('^recent\\s*$', re.I))
def recents(message, session):
    if False:
        for i in range(10):
            print('nop')
    articles = blogapi.get_recent_articles()
    if articles:
        reply = convert_to_article_reply(articles, message)
        return reply
    else:
        return '暂时还没有文章'

@robot.filter(re.compile('^help$', re.I))
def help(message, session):
    if False:
        print('Hello World!')
    return '欢迎关注!\n            默认会与图灵机器人聊天~~\n        你可以通过下面这些命令来获得信息\n        ?关键字搜索文章.\n        如?python.\n        category获得文章分类目录及文章数.\n        category-***获得该分类目录文章\n        如category-python\n        recent获得最新文章\n        help获得帮助.\n        weather:获得天气\n        如weather:西安\n        idcard:获得身份证信息\n        如idcard:61048119xxxxxxxxxx\n        music:音乐搜索\n        如music:阴天快乐\n        PS:以上标点符号都不支持中文标点~~\n        '

@robot.filter(re.compile('^weather\\:.*$', re.I))
def weather(message, session):
    if False:
        i = 10
        return i + 15
    return '建设中...'

@robot.filter(re.compile('^idcard\\:.*$', re.I))
def idcard(message, session):
    if False:
        print('Hello World!')
    return '建设中...'

@robot.handler
def echo(message, session):
    if False:
        return 10
    handler = MessageHandler(message, session)
    return handler.handler()

class MessageHandler:

    def __init__(self, message, session):
        if False:
            i = 10
            return i + 15
        userid = message.source
        self.message = message
        self.session = session
        self.userid = userid
        try:
            info = session[userid]
            self.userinfo = jsonpickle.decode(info)
        except Exception as e:
            userinfo = WxUserInfo()
            self.userinfo = userinfo

    @property
    def is_admin(self):
        if False:
            return 10
        return self.userinfo.isAdmin

    @property
    def is_password_set(self):
        if False:
            while True:
                i = 10
        return self.userinfo.isPasswordSet

    def save_session(self):
        if False:
            return 10
        info = jsonpickle.encode(self.userinfo)
        self.session[self.userid] = info

    def handler(self):
        if False:
            print('Hello World!')
        info = self.message.content
        if self.userinfo.isAdmin and info.upper() == 'EXIT':
            self.userinfo = WxUserInfo()
            self.save_session()
            return '退出成功'
        if info.upper() == 'ADMIN':
            self.userinfo.isAdmin = True
            self.save_session()
            return '输入管理员密码'
        if self.userinfo.isAdmin and (not self.userinfo.isPasswordSet):
            passwd = settings.WXADMIN
            if settings.TESTING:
                passwd = '123'
            if passwd.upper() == get_sha256(get_sha256(info)).upper():
                self.userinfo.isPasswordSet = True
                self.save_session()
                return '验证通过,请输入命令或者要执行的命令代码:输入helpme获得帮助'
            else:
                if self.userinfo.Count >= 3:
                    self.userinfo = WxUserInfo()
                    self.save_session()
                    return '超过验证次数'
                self.userinfo.Count += 1
                self.save_session()
                return '验证失败，请重新输入管理员密码:'
        if self.userinfo.isAdmin and self.userinfo.isPasswordSet:
            if self.userinfo.Command != '' and info.upper() == 'Y':
                return cmd_handler.run(self.userinfo.Command)
            else:
                if info.upper() == 'HELPME':
                    return cmd_handler.get_help()
                self.userinfo.Command = info
                self.save_session()
                return '确认执行: ' + info + ' 命令?'
        return ChatGPT.chat(info)

class WxUserInfo:

    def __init__(self):
        if False:
            print('Hello World!')
        self.isAdmin = False
        self.isPasswordSet = False
        self.Count = 0
        self.Command = ''