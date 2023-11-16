"""
@project: PyCharm
@file: pyqt_gui.py
@author: Shengqiang Zhang
@time: 2022-03-16 21:35:46
@mail: sqzhang77@gmail.com
"""
from wereader import *
from excel_func import *
import sys
import os
import time
from tqdm import tqdm
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QUrl
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineProfile
HEADERS = {'Host': 'i.weread.qq.com', 'Connection': 'keep-alive', 'Upgrade-Insecure-Requests': '1', 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36', 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3', 'Accept-Encoding': 'gzip, deflate, br', 'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'}
USER_VID = 0

class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self.DomainCookies = {}
        self.setWindowTitle('微信读书助手')
        self.resize(900, 600)
        self.setWindowFlags(Qt.WindowMinimizeButtonHint)
        self.setFixedSize(self.width(), self.height())
        url = 'https://weread.qq.com/#login'
        self.browser = QWebEngineView()
        self.profile = QWebEngineProfile.defaultProfile()
        self.profile.cookieStore().deleteAllCookies()
        self.profile.cookieStore().cookieAdded.connect(self.onCookieAdd)
        self.browser.loadFinished.connect(self.onLoadFinished)
        self.browser.load(QUrl(url))
        self.setCentralWidget(self.browser)

    def onLoadFinished(self):
        if False:
            print('Hello World!')
        global USER_VID
        global HEADERS
        cookies = ['{}={};'.format(key, value) for (key, value) in self.DomainCookies.items()]
        cookies = ' '.join(cookies)
        HEADERS.update(Cookie=cookies)
        if login_success(HEADERS):
            print('登录微信读书成功!')
            if 'wr_vid' in self.DomainCookies.keys():
                USER_VID = self.DomainCookies['wr_vid']
                print('用户id:{}'.format(USER_VID))
                self.browser.page().runJavaScript('alert("登录成功！")')
                self.close()
        else:
            print('请扫描二维码登录微信读书...')

    def onCookieAdd(self, cookie):
        if False:
            print('Hello World!')
        if 'weread.qq.com' in cookie.domain():
            name = cookie.name().data().decode('utf-8')
            value = cookie.value().data().decode('utf-8')
            if name not in self.DomainCookies:
                self.DomainCookies.update({name: value})

    def closeEvent(self, event):
        if False:
            print('Hello World!')
        '\n        重写closeEvent方法，实现窗体关闭时执行一些代码\n        :param event: close()触发的事件\n        :return: None\n        '
        self.setWindowTitle('退出中……')
        self.profile.cookieStore().deleteAllCookies()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
    data_dir = './导出资料/'
    note_dir = data_dir + '我的笔记/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(note_dir):
        os.makedirs(note_dir)
    books = get_bookshelf(USER_VID, HEADERS)
    books_finish_read = books['finishReadBooks']
    books_finish_read = [[book.bookId, book.title, book.author, book.cover] for book in books_finish_read]
    books_recent_read = books['recentBooks']
    books_recent_read = [[book.bookId, book.title, book.author, book.cover] for book in books_recent_read]
    books_all = books['allBooks']
    books_all = [[book.bookId, book.title, book.author, book.cover] for book in books_all]
    write_excel_xls(data_dir + '我的书架.xls', ['已读完的书籍', '最近阅读的书籍', '所有的书籍'], [['ID', '标题', '作者', '封面']])
    write_excel_xls_append(data_dir + '我的书架.xls', '已读完的书籍', books_finish_read)
    write_excel_xls_append(data_dir + '我的书架.xls', '最近阅读的书籍', books_recent_read)
    write_excel_xls_append(data_dir + '我的书架.xls', '所有的书籍', books_all)
    pbar = tqdm(books_finish_read)
    for book in pbar:
        book_id = book[0]
        book_name = book[1]
        for try_count in range(4):
            try:
                pbar.set_description('正在导出笔记【{}】'.format(book_name))
                notes = get_bookmarklist(book[0], HEADERS)
                with open(note_dir + book_name + '.txt', 'w', encoding='utf-8') as f:
                    f.write(notes)
                break
            except:
                pbar.set_description('获取笔记【{}】失败，开始第{}次重试'.format(book_name, try_count + 1))
                time.sleep(3)