from PIL import Image, ImageTk
from get_movie_data import movieData
from get_movie_data import get_url_data_in_ranking_list
from get_movie_data import get_url_data_in_keyWord
from tkinter import Tk
from tkinter import ttk
from tkinter import font
from tkinter import LabelFrame
from tkinter import Label
from tkinter import StringVar
from tkinter import Entry
from tkinter import END
from tkinter import Button
from tkinter import Frame
from tkinter import RIGHT
from tkinter import NSEW
from tkinter import NS
from tkinter import NW
from tkinter import N
from tkinter import Y
from tkinter import messagebox
from tkinter import DISABLED
from tkinter import NORMAL
from re import findall
from json import loads
from ssl import _create_unverified_context
from threading import Thread
from urllib.parse import quote
from webbrowser import open
import urllib
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def thread_it(func, *args):
    if False:
        while True:
            i = 10
    '\n    将函数打包进线程\n    '
    t = Thread(target=func, args=args)
    t.setDaemon(True)
    t.start()

def handlerAdaptor(fun, **kwds):
    if False:
        while True:
            i = 10
    '事件处理函数的适配器，相当于中介，那个event是从那里来的呢，我也纳闷，这也许就是python的伟大之处吧'
    return lambda event, fun=fun, kwds=kwds: fun(event, **kwds)

def save_img(img_url, file_name, file_path):
    if False:
        return 10
    '\n    下载指定url的图片，并保存运行目录下的img文件夹\n    :param img_url: 图片地址\n    :param file_name: 图片名字\n    :param file_path: 存储目录\n    :return:\n    '
    try:
        if not os.path.exists(file_path):
            print('文件夹', file_path, '不存在，重新建立')
            os.makedirs(file_path)
        file_suffix = os.path.splitext(img_url)[1]
        filename = '{}{}{}{}'.format(file_path, os.sep, file_name, file_suffix)
        if not os.path.exists(filename):
            print('文件', filename, '不存在，重新建立')
            urllib.request.urlretrieve(img_url, filename=filename)
        return filename
    except IOError as e:
        print('下载图片操作失败', e)
    except Exception as e:
        print('错误:', e)

def resize(w_box, h_box, pil_image):
    if False:
        while True:
            i = 10
    '\n    等比例缩放图片,并且限制在指定方框内\n    :param w_box,h_box: 指定方框的宽度和高度\n    :param pil_image: 原始图片\n    :return:\n    '
    f1 = 1.0 * w_box / pil_image.size[0]
    f2 = 1.0 * h_box / pil_image.size[1]
    factor = min([f1, f2])
    width = int(pil_image.size[0] * factor)
    height = int(pil_image.size[1] * factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)

def get_mid_str(content, startStr, endStr):
    if False:
        print('Hello World!')
    startIndex = content.find(startStr, 0)
    if startIndex >= 0:
        startIndex += len(startStr)
    else:
        return ''
    endIndex = content.find(endStr, startIndex)
    if endIndex >= 0 and endIndex >= startIndex:
        return content[startIndex:endIndex]
    else:
        return ''

class uiObject:

    def __init__(self):
        if False:
            return 10
        self.jsonData = ''
        self.jsonData_keyword = ''

    def show_GUI_movie_detail(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        显示 影片详情 界面GUI\n        '
        self.label_img['state'] = NORMAL
        self.label_movie_name['state'] = NORMAL
        self.label_movie_rating['state'] = NORMAL
        self.label_movie_time['state'] = NORMAL
        self.label_movie_type['state'] = NORMAL
        self.label_movie_actor['state'] = NORMAL

    def hidden_GUI_movie_detail(self):
        if False:
            return 10
        '\n        显示 影片详情 界面GUI\n        '
        self.label_img['state'] = DISABLED
        self.label_movie_name['state'] = DISABLED
        self.label_movie_rating['state'] = DISABLED
        self.label_movie_time['state'] = DISABLED
        self.label_movie_type['state'] = DISABLED
        self.label_movie_actor['state'] = DISABLED

    def show_IDMB_rating(self):
        if False:
            return 10
        '\n        显示IDM评分\n        '
        self.label_movie_rating_imdb.config(text='正在加载IMDB评分')
        self.B_0_imdb['state'] = DISABLED
        item = self.treeview.selection()
        if item:
            item_text = self.treeview.item(item, 'values')
            movieName = item_text[0]
            for movie in self.jsonData:
                if movie['title'] == movieName:
                    context = _create_unverified_context()
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36'}
                    req = urllib.request.Request(url=movie['url'], headers=headers)
                    f = urllib.request.urlopen(req, context=context)
                    response = f.read().decode()
                    self.clear_tree(self.treeview_play_online)
                    s = response
                    name = findall('<a class="playBtn" data-cn="(.*?)" data-impression-track', s)
                    down_url = findall('data-cn=".*?" href="(.*?)" target=', s)
                    res_list = []
                    for i in range(len(name)):
                        res_list.append([name[i], '限VIP免费', down_url[i]])
                    self.add_tree(res_list, self.treeview_play_online)
                    self.clear_tree(self.treeview_save_cloud_disk)
                    res_list = []
                    res_list.append(['56网盘搜索', '有效', 'https://www.56wangpan.com/search/o2kw' + quote(movie['title'])])
                    res_list.append(['爱搜资源', '有效', 'https://www.aisouziyuan.com/?name=' + quote(movie['title']) + '&page=1'])
                    res_list.append(['盘多多', '有效', 'http://www.panduoduo.net/s/comb/n-' + quote(movie['title']) + '&f-f4'])
                    res_list.append(['小白盘', '有效', 'https://www.xiaobaipan.com/list-' + quote(movie['title']) + '-1.html'])
                    res_list.append(['云盘精灵', '有效', 'https://www.yunpanjingling.com/search/' + quote(movie['title']) + '?sort=size.desc'])
                    self.add_tree(res_list, self.treeview_save_cloud_disk)
                    self.clear_tree(self.treeview_bt_download)
                    res_list = []
                    res_list.append(['19影视', '有效', 'https://www.19kan.com/vodsearch.html?wd=' + quote(movie['title'])])
                    res_list.append(['2TU影院', '有效', 'http://www.82tu.cc/search.php?submit=%E6%90%9C+%E7%B4%A2&searchword=' + quote(movie['title'])])
                    res_list.append(['4K电影', '有效', 'https://www.dygc.org/?s=' + quote(movie['title'])])
                    res_list.append(['52 Movie', '有效', 'http://www.52movieba.com/search.htm?keyword=' + quote(movie['title'])])
                    res_list.append(['592美剧', '有效', 'http://www.592meiju.com/search/?wd=' + quote(movie['title'])])
                    res_list.append(['97电影网', '有效', 'http://www.55xia.com/search?q=' + quote(movie['title'])])
                    res_list.append(['98TVS', '有效', 'http://www.98tvs.com/?s=' + quote(movie['title'])])
                    res_list.append(['9去这里', '有效', 'http://9qzl.com/index.php?s=/video/search/wd/' + quote(movie['title'])])
                    res_list.append(['CK电影', '有效', 'http://www.ck180.net/search.html?q=' + quote(movie['title'])])
                    res_list.append(['LOL电影', '有效', 'http://www.993dy.com/index.php?m=vod-search&wd=' + quote(movie['title'])])
                    res_list.append(['MP4Vv', '有效', 'http://www.mp4pa.com/search.php?searchword=' + quote(movie['title'])])
                    res_list.append(['MP4电影', '有效', 'http://www.domp4.com/search/' + quote(movie['title']) + '-1.html'])
                    res_list.append(['TL95', '有效', 'http://www.tl95.com/?s=' + quote(movie['title'])])
                    res_list.append(['比特大雄', '有效', 'https://www.btdx8.com/?s=' + quote(movie['title'])])
                    res_list.append(['比特影视', '有效', 'https://www.bteye.com/search/' + quote(movie['title'])])
                    res_list.append(['春晓影视', '有效', 'http://search.chunxiao.tv/?keyword=' + quote(movie['title'])])
                    res_list.append(['第一电影网', '有效', 'https://www.001d.com/?s=' + quote(movie['title'])])
                    res_list.append(['电影日志', '有效', 'http://www.dyrizhi.com/search?s=' + quote(movie['title'])])
                    res_list.append(['高清888', '有效', 'https://www.gaoqing888.com/search?kw=' + quote(movie['title'])])
                    res_list.append(['高清MP4', '有效', 'http://www.mp4ba.com/index.php?m=vod-search&wd=' + quote(movie['title'])])
                    res_list.append(['高清电台', '有效', 'https://gaoqing.fm/s.php?q=' + quote(movie['title'])])
                    res_list.append(['高清控', '有效', 'http://www.gaoqingkong.com/?s=' + quote(movie['title'])])
                    res_list.append(['界绍部', '有效', 'http://www.jsb456.com/?s=' + quote(movie['title'])])
                    res_list.append(['看美剧', '有效', 'http://www.kanmeiju.net/index.php?s=/video/search/wd/' + quote(movie['title'])])
                    res_list.append(['蓝光网', '有效', 'http://www.languang.co/?s=' + quote(movie['title'])])
                    res_list.append(['老司机电影', '有效', 'http://www.lsjdyw.net/search/?s=' + quote(movie['title'])])
                    res_list.append(['乐赏电影', '有效', 'http://www.gscq.me/search.htm?keyword=' + quote(movie['title'])])
                    res_list.append(['美剧汇', '有效', 'http://www.meijuhui.net/search.php?q=' + quote(movie['title'])])
                    res_list.append(['美剧鸟', '有效', 'http://www.meijuniao.com/index.php?s=vod-search-wd-' + quote(movie['title'])])
                    res_list.append(['迷你MP4', '有效', 'http://www.minimp4.com/search?q=' + quote(movie['title'])])
                    res_list.append(['泡饭影视', '有效', 'http://www.chapaofan.com/search/' + quote(movie['title'])])
                    res_list.append(['片吧', '有效', 'http://so.pianbar.com/search.aspx?q=' + quote(movie['title'])])
                    res_list.append(['片源网', '有效', 'http://pianyuan.net/search?q=' + quote(movie['title'])])
                    res_list.append(['飘花资源网', '有效', 'https://www.piaohua.com/plus/search.php?kwtype=0&keyword=' + quote(movie['title'])])
                    res_list.append(['趣味源', '有效', 'http://quweiyuan.cc/?s=' + quote(movie['title'])])
                    res_list.append(['人生05', '有效', 'http://www.rs05.com/search.php?s=' + quote(movie['title'])])
                    res_list.append(['贪玩影视', '有效', 'http://www.tanwanyingshi.com/movie/search?keyword=' + quote(movie['title'])])
                    res_list.append(['新片网', '有效', 'http://www.91xinpian.com/index.php?m=vod-search&wd=' + quote(movie['title'])])
                    res_list.append(['迅雷影天堂', '有效', 'https://www.xl720.com/?s=' + quote(movie['title'])])
                    res_list.append(['迅影网', '有效', 'http://www.xunyingwang.com/search?q=' + quote(movie['title'])])
                    res_list.append(['一只大榴莲', '有效', 'http://www.llduang.com/?s=' + quote(movie['title'])])
                    res_list.append(['音范丝', '有效', 'http://www.yinfans.com/?s=' + quote(movie['title'])])
                    res_list.append(['影海', '有效', 'http://www.yinghub.com/search/list.html?keyword=' + quote(movie['title'])])
                    res_list.append(['影视看看', '有效', 'http://www.yskk.tv/index.php?m=vod-search&wd=' + quote(movie['title'])])
                    res_list.append(['云播网', '有效', 'http://www.yunbowang.cn/index.php?m=vod-search&wd=' + quote(movie['title'])])
                    res_list.append(['中国高清网', '有效', 'http://gaoqing.la/?s=' + quote(movie['title'])])
                    res_list.append(['最新影视站', '有效', 'http://www.zxysz.com/?s=' + quote(movie['title'])])
                    self.add_tree(res_list, self.treeview_bt_download)
                    imdb_num = get_mid_str(response, 'IMDb:</span>', '<br>').strip()
                    imdb_url = 'https://www.imdb.com/title/{}/'.format(imdb_num)
                    print('电影名:{}, IMDb:{}'.format(movie['title'], imdb_num))
                    f = urllib.request.urlopen(imdb_url)
                    data_imdb = f.read().decode()
                    rating_imdb = get_mid_str(data_imdb, '{"@type":"AggregateRating"', '}')
                    rating_imdb = rating_imdb.split(':')[-1]
                    self.label_movie_rating_imdb.config(text='IMDB评分:' + rating_imdb + '分')
        self.B_0_imdb['state'] = NORMAL

    def project_statement_show(self, event):
        if False:
            return 10
        open('https://github.com/shengqiangzhang/examples-of-web-crawlers')

    def project_statement_get_focus(self, event):
        if False:
            for i in range(10):
                print('nop')
        self.project_statement.config(fg='blue', cursor='hand1')

    def project_statement_lose_focus(self, event):
        if False:
            i = 10
            return i + 15
        self.project_statement.config(fg='#FF0000')

    def show_movie_data(self, event):
        if False:
            print('Hello World!')
        '\n        显示某个被选择的电影的详情信息\n        '
        self.B_0_imdb['state'] = NORMAL
        self.label_movie_rating_imdb.config(text='IMDB评分')
        self.clear_tree(self.treeview_play_online)
        self.clear_tree(self.treeview_save_cloud_disk)
        self.clear_tree(self.treeview_bt_download)
        item = self.treeview.selection()
        if item:
            item_text = self.treeview.item(item, 'values')
            movieName = item_text[0]
            for movie in self.jsonData:
                if movie['title'] == movieName:
                    img_url = movie['cover_url']
                    movie_name = movie['title']
                    file_name = save_img(img_url, movie_name, 'img')
                    self.show_movie_img(file_name)
                    self.label_movie_name.config(text=movie['title'])
                    if isinstance(movie['actors'], list):
                        string_actors = '、'.join(movie['actors'])
                    else:
                        string_actors = movie['actors']
                    self.label_movie_actor.config(text=string_actors)
                    self.label_movie_rating.config(text=str(movie['rating'][0]) + '分 ' + str(movie['vote_count']) + '人评价')
                    self.label_movie_time.config(text=movie['release_date'])
                    self.label_movie_type.config(text=movie['types'])
                    break

    def show_movie_img(self, file_name):
        if False:
            while True:
                i = 10
        '\n        更新图片GUI\n        :param file_name: 图片路径\n        :return:\n        '
        img_open = Image.open(file_name)
        pil_image_resized = resize(160, 230, img_open)
        img = ImageTk.PhotoImage(pil_image_resized)
        self.label_img.config(image=img, width=pil_image_resized.size[0], height=pil_image_resized.size[1])
        self.label_img.image = img

    def center_window(self, root, w, h):
        if False:
            print('Hello World!')
        '\n        窗口居于屏幕中央\n        :param root: root\n        :param w: 窗口宽度\n        :param h: 窗口高度\n        :return:\n        '
        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()
        x = ws / 2 - w / 2
        y = hs / 2 - h / 2
        root.geometry('%dx%d+%d+%d' % (w, h, x, y))

    def clear_tree(self, tree):
        if False:
            for i in range(10):
                print('nop')
        '\n        清空表格\n        '
        x = tree.get_children()
        for item in x:
            tree.delete(item)

    def add_tree(self, list, tree):
        if False:
            while True:
                i = 10
        '\n        新增数据到表格\n        '
        i = 0
        for subList in list:
            tree.insert('', 'end', values=subList)
            i = i + 1
        tree.grid()

    def searh_movie_in_rating(self):
        if False:
            i = 10
            return i + 15
        '\n        从排行榜中搜索符合条件的影片信息\n        '
        self.clear_tree(self.treeview)
        self.B_0['state'] = DISABLED
        self.C_type['state'] = DISABLED
        self.T_count['state'] = DISABLED
        self.T_rating['state'] = DISABLED
        self.T_vote['state'] = DISABLED
        self.B_0_keyword['state'] = DISABLED
        self.T_vote_keyword['state'] = DISABLED
        self.B_0['text'] = '正在努力搜索'
        self.jsonData = ''
        jsonMovieData = loads(movieData)
        for subMovieData in jsonMovieData:
            if subMovieData['title'] == self.C_type.get():
                res_data = get_url_data_in_ranking_list(subMovieData['type'], self.T_count.get(), self.T_rating.get(), self.T_vote.get())
                if len(res_data) == 2:
                    res_list = res_data[0]
                    jsonData = res_data[1]
                    self.jsonData = jsonData
                    self.add_tree(res_list, self.treeview)
                else:
                    err_str = res_data[0]
                    messagebox.showinfo('提示', err_str[:1000])
        self.B_0['state'] = NORMAL
        self.C_type['state'] = 'readonly'
        self.T_count['state'] = NORMAL
        self.T_rating['state'] = NORMAL
        self.T_vote['state'] = NORMAL
        self.B_0_keyword['state'] = NORMAL
        self.T_vote_keyword['state'] = NORMAL
        self.B_0['text'] = '从排行榜搜索'

    def keyboard_T_vote_keyword(self, event):
        if False:
            for i in range(10):
                print('nop')
        '\n        在搜索框中键入回车键后触发相应的事件\n        :param event:\n        :return:\n        '
        thread_it(self.searh_movie_in_keyword)

    def searh_movie_in_keyword(self):
        if False:
            while True:
                i = 10
        '\n        从关键字中搜索符合条件的影片信息\n        '
        self.clear_tree(self.treeview)
        self.B_0['state'] = DISABLED
        self.C_type['state'] = DISABLED
        self.T_count['state'] = DISABLED
        self.T_rating['state'] = DISABLED
        self.T_vote['state'] = DISABLED
        self.B_0_keyword['state'] = DISABLED
        self.T_vote_keyword['state'] = DISABLED
        self.B_0_keyword['text'] = '正在努力搜索'
        self.jsonData = ''
        res_data = get_url_data_in_keyWord(self.T_vote_keyword.get())
        if len(res_data) == 2:
            res_list = res_data[0]
            jsonData = res_data[1]
            self.jsonData = jsonData
            self.add_tree(res_list, self.treeview)
        else:
            err_str = res_data[0]
            messagebox.showinfo('提示', err_str[:1000])
        self.B_0['state'] = NORMAL
        self.C_type['state'] = 'readonly'
        self.T_count['state'] = NORMAL
        self.T_rating['state'] = NORMAL
        self.T_vote['state'] = NORMAL
        self.B_0_keyword['state'] = NORMAL
        self.T_vote_keyword['state'] = NORMAL
        self.B_0_keyword['text'] = '从关键字搜索'

    def open_in_browser_douban_url(self, event):
        if False:
            return 10
        '\n        从浏览器中打开指定网页\n        :param\n        :return:\n        '
        item = self.treeview.selection()
        if item:
            item_text = self.treeview.item(item, 'values')
            movieName = item_text[0]
            for movie in self.jsonData:
                if movie['title'] == movieName:
                    open(movie['url'])

    def open_in_browser(self, event):
        if False:
            print('Hello World!')
        '\n        从浏览器中打开指定网页\n        :param\n        :return:\n        '
        item = self.treeview_play_online.selection()
        if item:
            item_text = self.treeview_play_online.item(item, 'values')
            url = item_text[2]
            open(url)

    def open_in_browser_cloud_disk(self, event):
        if False:
            for i in range(10):
                print('nop')
        '\n        从浏览器中打开指定网页\n        :param\n        :return:\n        '
        item = self.treeview_save_cloud_disk.selection()
        if item:
            item_text = self.treeview_save_cloud_disk.item(item, 'values')
            url = item_text[2]
            open(url)

    def open_in_browser_bt_download(self, event):
        if False:
            i = 10
            return i + 15
        '\n        从浏览器中打开指定网页\n        :param\n        :return:\n        '
        item = self.treeview_bt_download.selection()
        if item:
            item_text = self.treeview_bt_download.item(item, 'values')
            url = item_text[2]
            open(url)

    def ui_process(self):
        if False:
            i = 10
            return i + 15
        '\n        Ui主程序\n        :param\n        :return:\n        '
        root = Tk()
        self.root = root
        root.title('豆瓣电影小助手(可筛选、下载自定义电影)')
        self.center_window(root, 1000, 565)
        root.resizable(0, 0)
        labelframe = LabelFrame(root, width=660, height=300, text='搜索电影')
        labelframe.place(x=5, y=5)
        self.labelframe = labelframe
        L_typeId = Label(labelframe, text='电影类型')
        L_typeId.place(x=0, y=10)
        self.L_typeId = L_typeId
        comvalue = StringVar()
        C_type = ttk.Combobox(labelframe, width=5, textvariable=comvalue, state='readonly')
        jsonMovieData = loads(movieData)
        movieList = []
        for subMovieData in jsonMovieData:
            movieList.append(subMovieData['title'])
        C_type['values'] = movieList
        C_type.current(9)
        C_type.place(x=65, y=8)
        self.C_type = C_type
        L_count = Label(labelframe, text='获取数量=')
        L_count.place(x=150, y=10)
        self.L_count = L_count
        T_count = Entry(labelframe, width=5)
        T_count.delete(0, END)
        T_count.insert(0, '100')
        T_count.place(x=220, y=7)
        self.T_count = T_count
        L_rating = Label(labelframe, text='影片评分>')
        L_rating.place(x=280, y=10)
        self.L_rating = L_rating
        T_rating = Entry(labelframe, width=5)
        T_rating.delete(0, END)
        T_rating.insert(0, '8.0')
        T_rating.place(x=350, y=7)
        self.T_rating = T_rating
        L_vote = Label(labelframe, text='评价人数>')
        L_vote.place(x=410, y=10)
        self.L_vote = L_vote
        T_vote = Entry(labelframe, width=7)
        T_vote.delete(0, END)
        T_vote.insert(0, '100000')
        T_vote.place(x=480, y=7)
        self.T_vote = T_vote
        B_0 = Button(labelframe, text='从排行榜搜索')
        B_0.place(x=560, y=10)
        self.B_0 = B_0
        frame_root = Frame(labelframe, width=400)
        frame_l = Frame(frame_root)
        frame_r = Frame(frame_root)
        self.frame_root = frame_root
        self.frame_l = frame_l
        self.frame_r = frame_r
        columns = ('影片名字', '影片评分', '同类排名', '评价人数')
        treeview = ttk.Treeview(frame_l, height=10, show='headings', columns=columns)
        treeview.column('影片名字', width=210, anchor='center')
        treeview.column('影片评分', width=210, anchor='center')
        treeview.column('同类排名', width=100, anchor='center')
        treeview.column('评价人数', width=100, anchor='center')
        treeview.heading('影片名字', text='影片名字')
        treeview.heading('影片评分', text='影片评分')
        treeview.heading('同类排名', text='同类排名')
        treeview.heading('评价人数', text='评价人数')
        vbar = ttk.Scrollbar(frame_r, command=treeview.yview)
        treeview.configure(yscrollcommand=vbar.set)
        treeview.pack()
        self.treeview = treeview
        vbar.pack(side=RIGHT, fill=Y)
        self.vbar = vbar
        frame_l.grid(row=0, column=0, sticky=NSEW)
        frame_r.grid(row=0, column=1, sticky=NS)
        frame_root.place(x=5, y=70)
        L_vote_keyword = Label(labelframe, text='影片名称')
        L_vote_keyword.place(x=0, y=40)
        self.L_vote_keyword = L_vote_keyword
        T_vote_keyword = Entry(labelframe, width=53)
        T_vote_keyword.delete(0, END)
        T_vote_keyword.insert(0, '我不是药神')
        T_vote_keyword.place(x=66, y=37)
        self.T_vote_keyword = T_vote_keyword
        B_0_keyword = Button(labelframe, text='从关键字搜索')
        B_0_keyword.place(x=560, y=40)
        self.B_0_keyword = B_0_keyword
        labelframe_movie_detail = LabelFrame(root, text='影片详情')
        labelframe_movie_detail.place(x=670, y=5)
        self.labelframe_movie_detail = labelframe_movie_detail
        frame_left_movie_detail = Frame(labelframe_movie_detail, width=160, height=280)
        frame_left_movie_detail.grid(row=0, column=0)
        self.frame_left_movie_detail = frame_left_movie_detail
        frame_right_movie_detail = Frame(labelframe_movie_detail, width=160, height=280)
        frame_right_movie_detail.grid(row=0, column=1)
        self.frame_right_movie_detail = frame_right_movie_detail
        label_img = Label(frame_left_movie_detail, text='', anchor=N)
        label_img.place(x=0, y=0)
        self.label_img = label_img
        ft_rating_imdb = font.Font(weight=font.BOLD)
        label_movie_rating_imdb = Label(frame_left_movie_detail, text='IMDB评分', fg='#7F00FF', font=ft_rating_imdb, anchor=NW)
        label_movie_rating_imdb.place(x=0, y=250)
        self.label_movie_rating_imdb = label_movie_rating_imdb
        B_0_imdb = Button(frame_left_movie_detail, text='初始化')
        B_0_imdb.place(x=115, y=250)
        self.B_0_imdb = B_0_imdb
        ft = font.Font(size=15, weight=font.BOLD)
        label_movie_name = Label(frame_right_movie_detail, text='影片名字', fg='#FF0000', font=ft, anchor=NW)
        label_movie_name.place(x=0, y=0)
        self.label_movie_name = label_movie_name
        ft_rating = font.Font(weight=font.BOLD)
        label_movie_rating = Label(frame_right_movie_detail, text='影片评价', fg='#7F00FF', font=ft_rating, anchor=NW)
        label_movie_rating.place(x=0, y=30)
        self.label_movie_rating = label_movie_rating
        ft_time = font.Font(weight=font.BOLD)
        label_movie_time = Label(frame_right_movie_detail, text='影片日期', fg='#666600', font=ft_time, anchor=NW)
        label_movie_time.place(x=0, y=60)
        self.label_movie_time = label_movie_time
        ft_type = font.Font(weight=font.BOLD)
        label_movie_type = Label(frame_right_movie_detail, text='影片类型', fg='#330033', font=ft_type, anchor=NW)
        label_movie_type.place(x=0, y=90)
        self.label_movie_type = label_movie_type
        label_movie_actor = Label(frame_right_movie_detail, text='影片演员', wraplength=135, justify='left', anchor=NW)
        label_movie_actor.place(x=0, y=120)
        self.label_movie_actor = label_movie_actor
        labelframe_movie_play_online = LabelFrame(root, width=324, height=230, text='在线观看')
        labelframe_movie_play_online.place(x=5, y=305)
        self.labelframe_movie_play_online = labelframe_movie_play_online
        frame_root_play_online = Frame(labelframe_movie_play_online, width=324)
        frame_l_play_online = Frame(frame_root_play_online)
        frame_r_play_online = Frame(frame_root_play_online)
        self.frame_root_play_online = frame_root_play_online
        self.frame_l_play_online = frame_l_play_online
        self.frame_r_play_online = frame_r_play_online
        columns_play_online = ('来源名称', '是否免费', '播放地址')
        treeview_play_online = ttk.Treeview(frame_l_play_online, height=10, show='headings', columns=columns_play_online)
        treeview_play_online.column('来源名称', width=90, anchor='center')
        treeview_play_online.column('是否免费', width=80, anchor='center')
        treeview_play_online.column('播放地址', width=120, anchor='center')
        treeview_play_online.heading('来源名称', text='来源名称')
        treeview_play_online.heading('是否免费', text='是否免费')
        treeview_play_online.heading('播放地址', text='播放地址')
        vbar_play_online = ttk.Scrollbar(frame_r_play_online, command=treeview_play_online.yview)
        treeview_play_online.configure(yscrollcommand=vbar_play_online.set)
        treeview_play_online.pack()
        self.treeview_play_online = treeview_play_online
        vbar_play_online.pack(side=RIGHT, fill=Y)
        self.vbar_play_online = vbar_play_online
        frame_l_play_online.grid(row=0, column=0, sticky=NSEW)
        frame_r_play_online.grid(row=0, column=1, sticky=NS)
        frame_root_play_online.place(x=5, y=0)
        labelframe_movie_save_cloud_disk = LabelFrame(root, width=324, height=230, text='云盘搜索')
        labelframe_movie_save_cloud_disk.place(x=340, y=305)
        self.labelframe_movie_save_cloud_disk = labelframe_movie_save_cloud_disk
        frame_root_save_cloud_disk = Frame(labelframe_movie_save_cloud_disk, width=324)
        frame_l_save_cloud_disk = Frame(frame_root_save_cloud_disk)
        frame_r_save_cloud_disk = Frame(frame_root_save_cloud_disk)
        self.frame_root_save_cloud_disk = frame_root_save_cloud_disk
        self.frame_l_save_cloud_disk = frame_l_save_cloud_disk
        self.frame_r_save_cloud_disk = frame_r_save_cloud_disk
        columns_save_cloud_disk = ('来源名称', '是否有效', '播放地址')
        treeview_save_cloud_disk = ttk.Treeview(frame_l_save_cloud_disk, height=10, show='headings', columns=columns_save_cloud_disk)
        treeview_save_cloud_disk.column('来源名称', width=90, anchor='center')
        treeview_save_cloud_disk.column('是否有效', width=80, anchor='center')
        treeview_save_cloud_disk.column('播放地址', width=120, anchor='center')
        treeview_save_cloud_disk.heading('来源名称', text='来源名称')
        treeview_save_cloud_disk.heading('是否有效', text='是否有效')
        treeview_save_cloud_disk.heading('播放地址', text='播放地址')
        vbar_save_cloud_disk = ttk.Scrollbar(frame_r_save_cloud_disk, command=treeview_save_cloud_disk.yview)
        treeview_save_cloud_disk.configure(yscrollcommand=vbar_save_cloud_disk.set)
        treeview_save_cloud_disk.pack()
        self.treeview_save_cloud_disk = treeview_save_cloud_disk
        vbar_save_cloud_disk.pack(side=RIGHT, fill=Y)
        self.vbar_save_cloud_disk = vbar_save_cloud_disk
        frame_l_save_cloud_disk.grid(row=0, column=0, sticky=NSEW)
        frame_r_save_cloud_disk.grid(row=0, column=1, sticky=NS)
        frame_root_save_cloud_disk.place(x=5, y=0)
        labelframe_movie_bt_download = LabelFrame(root, width=324, height=230, text='影视下载')
        labelframe_movie_bt_download.place(x=670, y=305)
        self.labelframe_movie_bt_download = labelframe_movie_bt_download
        frame_root_bt_download = Frame(labelframe_movie_bt_download, width=324)
        frame_l_bt_download = Frame(frame_root_bt_download)
        frame_r_bt_download = Frame(frame_root_bt_download)
        self.frame_root_bt_download = frame_root_bt_download
        self.frame_l_bt_download = frame_l_bt_download
        self.frame_r_bt_download = frame_r_bt_download
        columns_bt_download = ('来源名称', '是否有效', '播放地址')
        treeview_bt_download = ttk.Treeview(frame_l_bt_download, height=10, show='headings', columns=columns_bt_download)
        treeview_bt_download.column('来源名称', width=90, anchor='center')
        treeview_bt_download.column('是否有效', width=80, anchor='center')
        treeview_bt_download.column('播放地址', width=120, anchor='center')
        treeview_bt_download.heading('来源名称', text='来源名称')
        treeview_bt_download.heading('是否有效', text='是否有效')
        treeview_bt_download.heading('播放地址', text='播放地址')
        vbar_bt_download = ttk.Scrollbar(frame_r_bt_download, command=treeview_bt_download.yview)
        treeview_bt_download.configure(yscrollcommand=vbar_bt_download.set)
        treeview_bt_download.pack()
        self.treeview_bt_download = treeview_bt_download
        vbar_bt_download.pack(side=RIGHT, fill=Y)
        self.vbar_bt_download = vbar_bt_download
        frame_l_bt_download.grid(row=0, column=0, sticky=NSEW)
        frame_r_bt_download.grid(row=0, column=1, sticky=NS)
        frame_root_bt_download.place(x=5, y=0)
        ft = font.Font(size=14, weight=font.BOLD)
        project_statement = Label(root, text='1.鼠标双击可打开相应的链接, 2.点击初始化按钮后将显示完整信息', fg='#FF0000', font=ft, anchor=NW)
        project_statement.place(x=5, y=540)
        self.project_statement = project_statement
        treeview.bind('<<TreeviewSelect>>', self.show_movie_data)
        treeview.bind('<Double-1>', self.open_in_browser_douban_url)
        treeview_play_online.bind('<Double-1>', self.open_in_browser)
        treeview_save_cloud_disk.bind('<Double-1>', self.open_in_browser_cloud_disk)
        treeview_bt_download.bind('<Double-1>', self.open_in_browser_bt_download)
        B_0.configure(command=lambda : thread_it(self.searh_movie_in_rating))
        B_0_keyword.configure(command=lambda : thread_it(self.searh_movie_in_keyword))
        B_0_imdb.configure(command=lambda : thread_it(self.show_IDMB_rating))
        T_vote_keyword.bind('<Return>', handlerAdaptor(self.keyboard_T_vote_keyword))
        project_statement.bind('<ButtonPress-1>', self.project_statement_show)
        project_statement.bind('<Enter>', self.project_statement_get_focus)
        project_statement.bind('<Leave>', self.project_statement_lose_focus)
        root.mainloop()