import random
import re
import time
import base64
from urllib.request import urlretrieve
from bs4 import BeautifulSoup
import PIL.Image as image
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

def save_base64img(data_str, save_name):
    if False:
        return 10
    '\n    将 base64 数据转化为图片保存到指定位置\n    :param data_str: base64 数据，不包含类型\n    :param save_name: 保存的全路径\n    '
    img_data = base64.b64decode(data_str)
    file = open(save_name, 'wb')
    file.write(img_data)
    file.close()

def get_base64_by_canvas(driver, class_name, contain_type):
    if False:
        i = 10
        return i + 15
    '\n    将 canvas 标签内容转换为 base64 数据\n    :param driver: webdriver 对象\n    :param class_name: canvas 标签的类名\n    :param contain_type: 返回的数据是否包含类型\n    :return: base64 数据\n    '
    bg_img = ''
    while len(bg_img) < 5000:
        getImgJS = 'return document.getElementsByClassName("' + class_name + '")[0].toDataURL("image/png");'
        bg_img = driver.execute_script(getImgJS)
        time.sleep(0.5)
    if contain_type:
        return bg_img
    else:
        return bg_img[bg_img.find(',') + 1:]

def save_bg(driver, bg_path='bg.png', bg_class='geetest_canvas_bg geetest_absolute'):
    if False:
        for i in range(10):
            print('nop')
    '\n    保存包含缺口的背景图\n    :param driver: webdriver 对象\n    :param bg_path: 保存路径\n    :param bg_class: 背景图的 class 属性\n    :return: 保存路径\n    '
    bg_img_data = get_base64_by_canvas(driver, bg_class, False)
    save_base64img(bg_img_data, bg_path)
    return bg_path

def save_full_bg(driver, full_bg_path='fbg.png', full_bg_class='geetest_canvas_fullbg geetest_fade geetest_absolute'):
    if False:
        while True:
            i = 10
    '\n    保存完整的的背景图\n    :param driver: webdriver 对象\n    :param full_bg_path: 保存路径\n    :param full_bg_class: 完整背景图的 class 属性\n    :return: 保存路径\n    '
    bg_img_data = get_base64_by_canvas(driver, full_bg_class, False)
    save_base64img(bg_img_data, full_bg_path)
    return full_bg_path

class Crack:

    def __init__(self, keyword):
        if False:
            print('Hello World!')
        self.url = '*'
        self.browser = webdriver.Chrome('D:\\chromedriver.exe')
        self.wait = WebDriverWait(self.browser, 100)
        self.keyword = keyword
        self.BORDER = 6

    def open(self):
        if False:
            for i in range(10):
                print('nop')
        '\n\t\t打开浏览器,并输入查询内容\n\t\t'
        self.browser.get(self.url)
        keyword = self.wait.until(EC.presence_of_element_located((By.ID, 'keyword_qycx')))
        bowton = self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'btn')))
        keyword.send_keys(self.keyword)
        bowton.click()

    def get_images(self, bg_filename='bg.jpg', fullbg_filename='fullbg.jpg'):
        if False:
            return 10
        '\n\t\t获取验证码图片\n\t\t:return: 图片的location信息\n\t\t'
        bg = []
        fullgb = []
        while bg == [] and fullgb == []:
            bf = BeautifulSoup(self.browser.page_source, 'lxml')
            bg = bf.find_all('div', class_='gt_cut_bg_slice')
            fullgb = bf.find_all('div', class_='gt_cut_fullbg_slice')
        bg_url = re.findall('url\\("(.*)"\\);', bg[0].get('style'))[0].replace('webp', 'jpg')
        fullgb_url = re.findall('url\\("(.*)"\\);', fullgb[0].get('style'))[0].replace('webp', 'jpg')
        bg_location_list = []
        fullbg_location_list = []
        for each_bg in bg:
            location = {}
            location['x'] = int(re.findall('background-position: (.*)px (.*)px;', each_bg.get('style'))[0][0])
            location['y'] = int(re.findall('background-position: (.*)px (.*)px;', each_bg.get('style'))[0][1])
            bg_location_list.append(location)
        for each_fullgb in fullgb:
            location = {}
            location['x'] = int(re.findall('background-position: (.*)px (.*)px;', each_fullgb.get('style'))[0][0])
            location['y'] = int(re.findall('background-position: (.*)px (.*)px;', each_fullgb.get('style'))[0][1])
            fullbg_location_list.append(location)
        urlretrieve(url=bg_url, filename=bg_filename)
        print('缺口图片下载完成')
        urlretrieve(url=fullgb_url, filename=fullbg_filename)
        print('背景图片下载完成')
        return (bg_location_list, fullbg_location_list)

    def get_merge_image(self, filename, location_list):
        if False:
            print('Hello World!')
        '\n\t\t根据位置对图片进行合并还原\n\t\t:filename:图片\n\t\t:location_list:图片位置\n\t\t'
        im = image.open(filename)
        new_im = image.new('RGB', (260, 116))
        im_list_upper = []
        im_list_down = []
        for location in location_list:
            if location['y'] == -58:
                im_list_upper.append(im.crop((abs(location['x']), 58, abs(location['x']) + 10, 166)))
            if location['y'] == 0:
                im_list_down.append(im.crop((abs(location['x']), 0, abs(location['x']) + 10, 58)))
        new_im = image.new('RGB', (260, 116))
        x_offset = 0
        for im in im_list_upper:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        x_offset = 0
        for im in im_list_down:
            new_im.paste(im, (x_offset, 58))
            x_offset += im.size[0]
        new_im.save(filename)
        return new_im

    def get_merge_image(self, filename, location_list):
        if False:
            print('Hello World!')
        '\n\t\t根据位置对图片进行合并还原\n\t\t:filename:图片\n\t\t:location_list:图片位置\n\t\t'
        im = image.open(filename)
        new_im = image.new('RGB', (260, 116))
        im_list_upper = []
        im_list_down = []
        for location in location_list:
            if location['y'] == -58:
                im_list_upper.append(im.crop((abs(location['x']), 58, abs(location['x']) + 10, 166)))
            if location['y'] == 0:
                im_list_down.append(im.crop((abs(location['x']), 0, abs(location['x']) + 10, 58)))
        new_im = image.new('RGB', (260, 116))
        x_offset = 0
        for im in im_list_upper:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        x_offset = 0
        for im in im_list_down:
            new_im.paste(im, (x_offset, 58))
            x_offset += im.size[0]
        new_im.save(filename)
        return new_im

    def is_pixel_equal(self, img1, img2, x, y):
        if False:
            while True:
                i = 10
        '\n\t\t判断两个像素是否相同\n\t\t:param image1: 图片1\n\t\t:param image2: 图片2\n\t\t:param x: 位置x\n\t\t:param y: 位置y\n\t\t:return: 像素是否相同\n\t\t'
        pix1 = img1.load()[x, y]
        pix2 = img2.load()[x, y]
        threshold = 60
        if abs(pix1[0] - pix2[0] < threshold) and abs(pix1[1] - pix2[1] < threshold) and abs(pix1[2] - pix2[2] < threshold):
            return True
        else:
            return False

    def get_gap(self, img1, img2):
        if False:
            print('Hello World!')
        '\n\t\t获取缺口偏移量\n\t\t:param img1: 不带缺口图片\n\t\t:param img2: 带缺口图片\n\t\t:return:\n\t\t'
        left = 43
        for i in range(left, img1.size[0]):
            for j in range(img1.size[1]):
                if not self.is_pixel_equal(img1, img2, i, j):
                    left = i
                    return left
        return left

    def get_track(self, distance):
        if False:
            i = 10
            return i + 15
        '\n\t\t根据偏移量获取移动轨迹\n\t\t:param distance: 偏移量\n\t\t:return: 移动轨迹\n\t\t'
        track = []
        current = 0
        mid = distance * 4 / 5
        t = 0.2
        v = 0
        while current < distance:
            if current < mid:
                a = 2
            else:
                a = -3
            v0 = v
            v = v0 + a * t
            move = v0 * t + 1 / 2 * a * t * t
            current += move
            track.append(round(move))
        return track

    def get_slider(self):
        if False:
            print('Hello World!')
        '\n\t\t获取滑块\n\t\t:return: 滑块对象\n\t\t'
        while True:
            try:
                slider = self.browser.find_element_by_xpath("//div[@class='gt_slider_knob gt_show']")
                break
            except:
                time.sleep(0.5)
        return slider

    def move_to_gap(self, slider, track):
        if False:
            for i in range(10):
                print('nop')
        '\n\t\t拖动滑块到缺口处\n\t\t:param slider: 滑块\n\t\t:param track: 轨迹\n\t\t:return:\n\t\t'
        ActionChains(self.browser).click_and_hold(slider).perform()
        while track:
            x = random.choice(track)
            ActionChains(self.browser).move_by_offset(xoffset=x, yoffset=0).perform()
            track.remove(x)
        time.sleep(0.5)
        ActionChains(self.browser).release().perform()

    def crack(self):
        if False:
            i = 10
            return i + 15
        self.open()
        bg_filename = 'bg.jpg'
        fullbg_filename = 'fullbg.jpg'
        (bg_location_list, fullbg_location_list) = self.get_images(bg_filename, fullbg_filename)
        bg_img = save_bg(self.browser)
        full_bg_img = save_full_bg(self.browser)
        gap = self.get_gap(image.open(full_bg_img), image.open(bg_img))
        print('缺口位置', gap)
        track = self.get_track(gap - self.BORDER)
        print('滑动滑块')
        print(track)
if __name__ == '__main__':
    print('开始验证')
    crack = Crack(u'中国移动')
    crack.crack()
    print('验证成功')