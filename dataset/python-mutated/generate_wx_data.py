from wxpy import *
from platform import system
from os.path import exists
from os import makedirs
from os import listdir
from shutil import rmtree
from queue import Queue
from threading import Thread
from time import sleep
from pyecharts.charts import Pie
from pyecharts.charts import Map
from pyecharts.charts import WordCloud
from pyecharts.charts import Bar
from pyecharts import options as opts
from requests import post
import PIL.Image as Image
import re
import random
import math
from cv2 import CascadeClassifier
from cv2 import imread
from cv2 import cvtColor
from cv2 import COLOR_BGR2GRAY
if 'Windows' in system():
    from os import startfile
    open_html = lambda x: startfile(x)
elif 'Darwin' in system():
    from subprocess import call
    open_html = lambda x: call(['open', x])
else:
    from subprocess import call
    open_html = lambda x: call(['xdg-open', x])

def sex_ratio():
    if False:
        i = 10
        return i + 15
    (male, female, other) = (0, 0, 0)
    for user in friends:
        if user.sex == 1:
            male += 1
        elif user.sex == 2:
            female += 1
        else:
            other += 1
    name_list = ['男性', '女性', '未设置']
    num_list = [male, female, other]
    pie = Pie()
    pie.add('微信好友性别比例', [list(z) for z in zip(name_list, num_list)])
    pie.set_global_opts(title_opts=opts.TitleOpts(title='微信好友性别比例'))
    pie.set_series_opts(label_opts=opts.LabelOpts(formatter='{b}: {c}'))
    pie.render('data/好友性别比例.html')

def region_distribution():
    if False:
        return 10
    province_dict = {'北京': 0, '上海': 0, '天津': 0, '重庆': 0, '河北': 0, '山西': 0, '吉林': 0, '辽宁': 0, '黑龙江': 0, '陕西': 0, '甘肃': 0, '青海': 0, '山东': 0, '福建': 0, '浙江': 0, '台湾': 0, '河南': 0, '湖北': 0, '湖南': 0, '江西': 0, '江苏': 0, '安徽': 0, '广东': 0, '海南': 0, '四川': 0, '贵州': 0, '云南': 0, '内蒙古': 0, '新疆': 0, '宁夏': 0, '广西': 0, '西藏': 0, '香港': 0, '澳门': 0}
    for user in friends:
        if user.province in province_dict:
            key = user.province
            province_dict[key] += 1
    province = list(province_dict.keys())
    values = list(province_dict.values())
    map = Map()
    map.add('微信好友地区分布', [list(z) for z in zip(province, values)], 'china')
    map.set_global_opts(title_opts=opts.TitleOpts(title='微信好友地区分布'), visualmap_opts=opts.VisualMapOpts())
    map.render(path='data/好友地区分布.html')
    max_count_province = ''
    for (key, value) in province_dict.items():
        if value == max(province_dict.values()):
            max_count_province = key
            break
    city_dict = {}
    for user in friends:
        if user.province == max_count_province:
            if user.city in city_dict.keys():
                city_dict[user.city] += 1
            else:
                city_dict[user.city] = 1
    bar = Bar()
    bar.add_xaxis([x for x in city_dict.keys()])
    bar.add_yaxis('地区分布', [x for x in city_dict.values()])
    bar.render('data/某省好友地区分布.html')

def statistics_friends():
    if False:
        i = 10
        return i + 15
    (unknown, known_male, known_female, known_other) = (0, 0, 0, 0)
    for user in friends:
        if user.remark_name.strip():
            if user.sex == 1:
                known_male += 1
            elif user.sex == 2:
                known_female += 1
            else:
                known_other += 1
        else:
            unknown += 1
    name_list = ['未设置备注的好友', '设置备注的男性好友', '设置备注的女性好友', '设置备注的其他好友']
    num_list = [unknown, known_male, known_female, known_other]
    pie = Pie()
    pie.add('你认识的好友比例', [list(z) for z in zip(name_list, num_list)])
    pie.set_global_opts(title_opts=opts.TitleOpts(title='你认识的好友比例'))
    pie.set_series_opts(label_opts=opts.LabelOpts(formatter='{b}: {c}'))
    pie.render('data/你认识的好友比例.html')

def analyze_remark_name():
    if False:
        while True:
            i = 10
    close_partner_dict = {'宝宝,猪,仙女,亲爱,老婆': 0, '老公': 0, '父亲,爸': 0, '母亲,妈': 0, '闺蜜,死党,基友': 0}
    for user in friends:
        for key in close_partner_dict.keys():
            name = key.split(',')
            for sub_name in name:
                if sub_name in user.remark_name:
                    close_partner_dict[key] += 1
                    break
    name_list = ['最重要的她', '最重要的他', '爸爸', '妈妈', '死党']
    num_list = [x for x in close_partner_dict.values()]
    pie = Pie()
    pie.add('可能是你最亲密的人', [list(z) for z in zip(name_list, num_list)])
    pie.set_global_opts(title_opts=opts.TitleOpts(title='可能是你最亲密的人'))
    pie.set_series_opts(label_opts=opts.LabelOpts(formatter='{b}: {c}'))
    pie.render('data/你最亲密的人.html')

def analyze_signature():
    if False:
        print('Hello World!')
    data = []
    for user in friends:
        new_signature = re.sub(re.compile('<span class.*?</span>', re.S), '', user.signature)
        if len(new_signature.split('\n')) == 1:
            data.append(new_signature)
    data = '\n'.join(data)
    postData = {'data': data, 'type': 'exportword', 'arg': '', 'beforeSend': 'undefined'}
    response = post('http://life.chacuo.net/convertexportword', data=postData)
    data = response.text.replace('{"status":1,"info":"ok","data":["', '')
    data = data.encode('utf-8').decode('unicode_escape')
    data = data.split('=====================================')[0]
    data = data.split('  ')
    stop_words_list = [',', '，', '、', 'the', 'a', 'is', '…', '·', 'э', 'д', 'э', 'м', 'ж', 'и', 'л', 'т', 'ы', 'н', 'з', 'м', '…', '…', '…', '…', '…', '、', '.', '。', '!', '！', ':', '：', '~', '|', '▽', '`', 'ノ', '♪', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', "'", '‘', '’', '“', '”', '的', '了', '是', '你', '我', '他', '她', '=', '\r', '\n', '\r\n', '\t', '以下关键词', '[', ']', '{', '}', '(', ')', '（', '）', 'span', '<', '>', 'class', 'html', '?', '就', '于', '下', '在', '吗', '嗯']
    tmp_data = []
    for word in data:
        if word not in stop_words_list:
            tmp_data.append(word)
    data = tmp_data
    signature_dict = {}
    for (index, word) in enumerate(data):
        print(u'正在统计好友签名数据，进度%d/%d，请耐心等待……' % (index + 1, len(data)))
        if word in signature_dict.keys():
            signature_dict[word] += 1
        else:
            signature_dict[word] = 1
    name = [x for x in signature_dict.keys()]
    value = [x for x in signature_dict.values()]
    wordcloud = WordCloud()
    wordcloud.add('微信好友个性签名词云图', [list(z) for z in zip(name, value)], word_size_range=[1, 100], shape='star')
    wordcloud.render('data/好友个性签名词云.html')

def download_head_image(thread_name):
    if False:
        print('Hello World!')
    while not queue_head_image.empty():
        user = queue_head_image.get()
        random_file_name = ''.join([str(random.randint(0, 9)) for x in range(15)])
        user.get_avatar(save_path='image/' + random_file_name + '.jpg')
        print(u'线程%d:正在下载微信好友头像数据，进度%d/%d，请耐心等待……' % (thread_name, len(friends) - queue_head_image.qsize(), len(friends)))

def generate_html(file_name):
    if False:
        while True:
            i = 10
    with open(file_name, 'w', encoding='utf-8') as f:
        data = '\n            <meta http-equiv=\'Content-Type\' content=\'text/html; charset=utf-8\'>\n            <meta charset="UTF-8">\n            <title>一键生成微信个人专属数据报告(了解你的微信社交历史)</title>\n            <meta name=\'keywords\' content=\'微信个人数据\'>\n            <meta name=\'description\' content=\'\'> \n\n            \n            <iframe name="iframe1" marginwidth=0 marginheight=0 width=100% height=60% src="data/好友地区分布.html" frameborder=0></iframe>\n            <iframe name="iframe2" marginwidth=0 marginheight=0 width=100% height=60% src="data/某省好友地区分布.html" frameborder=0></iframe>\n            <iframe name="iframe3" marginwidth=0 marginheight=0 width=100% height=60% src="data/好友性别比例.html" frameborder=0></iframe>\n            <iframe name="iframe4" marginwidth=0 marginheight=0 width=100% height=60% src="data/你认识的好友比例.html" frameborder=0></iframe>\n            <iframe name="iframe5" marginwidth=0 marginheight=0 width=100% height=60% src="data/你最亲密的人.html" frameborder=0></iframe>\n            <iframe name="iframe6" marginwidth=0 marginheight=0 width=100% height=60% src="data/特殊好友分析.html" frameborder=0></iframe>\n            <iframe name="iframe7" marginwidth=0 marginheight=0 width=100% height=60% src="data/共同所在群聊分析.html" frameborder=0></iframe>\n            <iframe name="iframe8" marginwidth=0 marginheight=0 width=100% height=60% src="data/好友个性签名词云.html" frameborder=0></iframe>\n            <iframe name="iframe9" marginwidth=0 marginheight=0 width=100% height=60% src="data/微信好友头像拼接图.html" frameborder=0></iframe>\n            <iframe name="iframe10" marginwidth=0 marginheight=0 width=100% height=60% src="data/使用人脸的微信好友头像拼接图.html" frameborder=0></iframe>\n        '
        f.write(data)

def init_folders():
    if False:
        print('Hello World!')
    if not exists('image'):
        makedirs('image')
    else:
        rmtree('image')
        makedirs('image')
    if not exists('data'):
        makedirs('data')
    else:
        rmtree('data')
        makedirs('data')

def merge_head_image():
    if False:
        i = 10
        return i + 15
    pics = listdir('image')
    numPic = len(pics)
    eachsize = int(math.sqrt(float(640 * 640) / numPic))
    numrow = int(640 / eachsize)
    numcol = int(numPic / numrow)
    toImage = Image.new('RGB', (eachsize * numrow, eachsize * numcol))
    x = 0
    y = 0
    for (index, i) in enumerate(pics):
        print(u'正在拼接微信好友头像数据，进度%d/%d，请耐心等待……' % (index + 1, len(pics)))
        try:
            img = Image.open('image/' + i)
        except IOError:
            print(u'Error: 没有找到文件或读取文件失败')
        else:
            img = img.resize((eachsize, eachsize), Image.ANTIALIAS)
            toImage.paste(img, (x * eachsize, y * eachsize))
            x += 1
            if x == numrow:
                x = 0
                y += 1
    toImage.save('data/拼接' + '.jpg')
    with open('data/微信好友头像拼接图.html', 'w', encoding='utf-8') as f:
        data = '\n            <!DOCTYPE html>\n            <html xmlns="http://www.w3.org/1999/xhtml">\n            <head>\n                  <meta http-equiv=\'Content-Type\' content=\'text/html; charset=utf-8\'>\n                  <meta charset="utf-8" /> \n                  <title>微信好友头像拼接图</title> \n            </head>\n            <body>\n                <p><font size=4px><strong>微信好友头像拼接图</strong></font></p>\n                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\n                <img src="拼接.jpg" />\n            </body>\n            </html>\n        '
        f.write(data)

def detect_human_face():
    if False:
        while True:
            i = 10
    pics = listdir('image')
    count_face_image = 0
    list_name_face_image = []
    face_cascade = CascadeClassifier('model/haarcascade_frontalface_default.xml')
    for (index, file_name) in enumerate(pics):
        print(u'正在进行人脸识别，进度%d/%d，请耐心等待……' % (index + 1, len(pics)))
        img = imread('image/' + file_name)
        if img is None:
            continue
        gray = cvtColor(img, COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            count_face_image += 1
            list_name_face_image.append(file_name)
    print(u'使用人脸的头像%d/%d' % (count_face_image, len(pics)))
    pics = list_name_face_image
    numPic = len(pics)
    eachsize = int(math.sqrt(float(640 * 640) / numPic))
    numrow = int(640 / eachsize)
    numcol = int(numPic / numrow)
    toImage = Image.new('RGB', (eachsize * numrow, eachsize * numcol))
    x = 0
    y = 0
    for (index, i) in enumerate(pics):
        print(u'正在拼接使用人脸的微信好友头像数据，进度%d/%d，请耐心等待……' % (index + 1, len(pics)))
        try:
            img = Image.open('image/' + i)
        except IOError:
            print(u'Error: 没有找到文件或读取文件失败')
        else:
            img = img.resize((eachsize, eachsize), Image.ANTIALIAS)
            toImage.paste(img, (x * eachsize, y * eachsize))
            x += 1
            if x == numrow:
                x = 0
                y += 1
    toImage.save('data/使用人脸的拼接' + '.jpg')
    with open('data/使用人脸的微信好友头像拼接图.html', 'w', encoding='utf-8') as f:
        data = '\n            <!DOCTYPE html>\n            <html xmlns="http://www.w3.org/1999/xhtml">\n            <head>\n                  <meta http-equiv=\'Content-Type\' content=\'text/html; charset=utf-8\'>\n                  <meta charset="utf-8" /> \n                  <title>使用人脸的微信好友头像拼接图</title> \n            </head>\n            <body>\n                <p><font size=4px><strong>描述内容</strong></font></p>\n                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\n                <img src="使用人脸的拼接.jpg" />\n            </body>\n            </html>\n        '
        data = data.replace('描述内容', '在{}个好友中，有{}个好友使用真实的人脸作为头像'.format(len(friends), count_face_image))
        f.write(data)

def analyze_special_friends():
    if False:
        print('Hello World!')
    (star_friends, hide_my_post_friends, hide_his_post_friends, sticky_on_top_friends, stranger_friends) = (0, 0, 0, 0, 0)
    for user in friends:
        if 'StarFriend' in user.raw.keys():
            if user.raw['StarFriend'] == 1:
                star_friends += 1
        else:
            stranger_friends += 1
        if user.raw['ContactFlag'] in [259, 33027, 65795]:
            hide_my_post_friends += 1
        if user.raw['ContactFlag'] in [66051, 65537, 65539, 65795]:
            hide_his_post_friends += 1
        if user.raw['ContactFlag'] in [2051]:
            sticky_on_top_friends += 1
        if user.raw['ContactFlag'] in [73731]:
            stranger_friends += 1
    bar = Bar()
    bar.add_xaxis(['星标', '不让他看我朋友圈', '不看他朋友圈', '消息置顶', '陌生人'])
    bar.add_yaxis('特殊好友分析', [star_friends, hide_my_post_friends, hide_his_post_friends, sticky_on_top_friends, stranger_friends])
    bar.render('data/特殊好友分析.html')

def group_common_in():
    if False:
        i = 10
        return i + 15
    groups = bot.groups()
    dict_common_in = {}
    for x in friends[1:]:
        for y in groups:
            if x in y:
                name = x.nick_name
                if x.remark_name and x.remark_name != '':
                    name = x.remark_name
                if name in dict_common_in.keys():
                    dict_common_in[name] += 1
                else:
                    dict_common_in[name] = 1
    n = 0
    if len(dict_common_in) > 5:
        n = 6
    elif len(dict_common_in) > 4:
        n = 5
    elif len(dict_common_in) > 3:
        n = 4
    elif len(dict_common_in) > 2:
        n = 3
    elif len(dict_common_in) > 1:
        n = 2
    elif len(dict_common_in) > 0:
        n = 1
    sort_list = sorted(dict_common_in.items(), key=lambda item: item[1], reverse=True)
    sort_list = sort_list[:n]
    bar = Bar()
    bar.add_xaxis([x[0] for x in sort_list])
    bar.add_yaxis('共同所在群聊分析', [x[1] for x in sort_list])
    bar.render('data/共同所在群聊分析.html')
if __name__ == '__main__':
    init_folders()
    if 'Windows' in system():
        bot = Bot()
    elif 'Darwin' in system():
        bot = Bot(cache_path=True)
    elif 'Linux' in system():
        bot = Bot(console_qr=2, cache_path=True)
    else:
        print(u'无法识别你的操作系统类型，请自己设置')
        exit()
    print(u'正在获取微信好友数据信息，请耐心等待……')
    friends = bot.friends(update=False)
    print(u'微信好友数据信息获取完毕\n')
    print(u'正在分析你的群聊，请耐心等待……')
    group_common_in()
    print(u'分析群聊完毕\n')
    print(u'正在获取微信好友头像信息，请耐心等待……')
    queue_head_image = Queue()
    for user in friends:
        queue_head_image.put(user)
    for i in range(1, 10):
        t = Thread(target=download_head_image, args=(i,))
        t.start()
    print(u'微信好友头像信息获取完毕\n')
    print(u'正在分析好友性别比例，请耐心等待……')
    sex_ratio()
    print(u'分析好友性别比例完毕\n')
    print(u'正在分析好友地区分布，请耐心等待……')
    region_distribution()
    print(u'分析好友地区分布完毕\n')
    print(u'正在统计你认识的好友，请耐心等待……')
    statistics_friends()
    print(u'统计你认识的好友完毕\n')
    print(u'正在分析你最亲密的人，请耐心等待……')
    analyze_remark_name()
    print(u'分析你最亲密的人完毕\n')
    print(u'正在分析你的特殊好友，请耐心等待……')
    analyze_special_friends()
    print(u'分析你的特殊好友完毕\n')
    print(u'正在分析你的好友的个性签名，请耐心等待……')
    analyze_signature()
    print(u'分析你的好友的个性签名完毕\n')
    while not queue_head_image.empty():
        sleep(1)
    print(u'正在拼接所有微信好友头像数据，请耐心等待……')
    merge_head_image()
    print(u'拼接所有微信好友头像数据完毕\n')
    print(u'正在检测使用人脸作为头像的好友数量，请耐心等待……')
    detect_human_face()
    print(u'检测使用人脸作为头像的好友数量完毕\n')
    print(u'所有数据获取完毕，正在生成微信个人数据报告，请耐心等待……')
    generate_html('微信个人数据报告.html')
    print(u'生成微信个人数据报告完毕，该文件为当前目录下的[微信个人数据报告.html]\n')
    print(u'已为你自动打开 微信个人数据报告.html')
    open_html('微信个人数据报告.html')