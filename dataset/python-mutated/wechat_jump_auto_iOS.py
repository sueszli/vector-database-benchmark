"""
# === 思路 ===
# 核心：每次落稳之后截图，根据截图算出棋子的坐标和下一个块顶面的中点坐标，
#      根据两个点的距离乘以一个时间系数获得长按的时间
# 识别棋子：靠棋子的颜色来识别位置，通过截图发现最下面一行大概是一条
           直线，就从上往下一行一行遍历，比较颜色（颜色用了一个区间来比较）
           找到最下面的那一行的所有点，然后求个中点，求好之后再让 Y 轴坐标
           减小棋子底盘的一半高度从而得到中心点的坐标
# 识别棋盘：靠底色和方块的色差来做，从分数之下的位置开始，一行一行扫描，
           由于圆形的块最顶上是一条线，方形的上面大概是一个点，所以就
           用类似识别棋子的做法多识别了几个点求中点，这时候得到了块中点的 X
           轴坐标，这时候假设现在棋子在当前块的中心，根据一个通过截图获取的
           固定的角度来推出中点的 Y 坐标
# 最后：根据两点的坐标算距离乘以系数来获取长按时间（似乎可以直接用 X 轴距离）
"""
import os
import shutil
import time
import math
import random
import json
from PIL import Image, ImageDraw
import wda
with open('config.json', 'r') as f:
    config = json.load(f)
under_game_score_y = config['under_game_score_y']
press_coefficient = config['press_coefficient']
piece_base_height_1_2 = config['piece_base_height_1_2']
piece_body_width = config['piece_body_width']
time_coefficient = config['press_coefficient']
swipe = config.get('swipe', {'x1': 320, 'y1': 410, 'x2': 320, 'y2': 410})
VERSION = '1.1.4'
c = wda.Client()
s = c.session()
screenshot_backup_dir = 'screenshot_backups/'
if not os.path.isdir(screenshot_backup_dir):
    os.mkdir(screenshot_backup_dir)

def pull_screenshot():
    if False:
        while True:
            i = 10
    c.screenshot('1.png')

def jump(distance):
    if False:
        while True:
            i = 10
    press_time = distance * time_coefficient / 1000
    print('press time: {}'.format(press_time))
    s.tap_hold(random.uniform(0, 320), random.uniform(64, 320), press_time)

def backup_screenshot(ts):
    if False:
        for i in range(10):
            print('nop')
    '\n    为了方便失败的时候 debug\n    '
    if not os.path.isdir(screenshot_backup_dir):
        os.mkdir(screenshot_backup_dir)
    shutil.copy('1.png', '{}{}.png'.format(screenshot_backup_dir, ts))

def save_debug_creenshot(ts, im, piece_x, piece_y, board_x, board_y):
    if False:
        print('Hello World!')
    draw = ImageDraw.Draw(im)
    draw.line((piece_x, piece_y) + (board_x, board_y), fill=2, width=3)
    draw.line((piece_x, 0, piece_x, im.size[1]), fill=(255, 0, 0))
    draw.line((0, piece_y, im.size[0], piece_y), fill=(255, 0, 0))
    draw.line((board_x, 0, board_x, im.size[1]), fill=(0, 0, 255))
    draw.line((0, board_y, im.size[0], board_y), fill=(0, 0, 255))
    draw.ellipse((piece_x - 10, piece_y - 10, piece_x + 10, piece_y + 10), fill=(255, 0, 0))
    draw.ellipse((board_x - 10, board_y - 10, board_x + 10, board_y + 10), fill=(0, 0, 255))
    del draw
    im.save('{}{}_d.png'.format(screenshot_backup_dir, ts))

def set_button_position(im):
    if False:
        i = 10
        return i + 15
    '\n    将swipe设置为 `再来一局` 按钮的位置\n    '
    global swipe_x1, swipe_y1, swipe_x2, swipe_y2
    (w, h) = im.size
    left = w / 2
    top = 1003 * (h / 1280.0) + 10
    (swipe_x1, swipe_y1, swipe_x2, swipe_y2) = (left, top, left, top)

def find_piece_and_board(im):
    if False:
        print('Hello World!')
    (w, h) = im.size
    print('size: {}, {}'.format(w, h))
    piece_x_sum = piece_x_c = piece_y_max = 0
    board_x = board_y = 0
    scan_x_border = int(w / 8)
    scan_start_y = 0
    im_pixel = im.load()
    for i in range(under_game_score_y, h, 50):
        last_pixel = im_pixel[0, i]
        for j in range(1, w):
            pixel = im_pixel[j, i]
            if pixel != last_pixel:
                scan_start_y = i - 50
                break
        if scan_start_y:
            break
    print('scan_start_y: ', scan_start_y)
    for i in range(scan_start_y, int(h * 2 / 3)):
        for j in range(scan_x_border, w - scan_x_border):
            pixel = im_pixel[j, i]
            if 50 < pixel[0] < 60 and 53 < pixel[1] < 63 and (95 < pixel[2] < 110):
                piece_x_sum += j
                piece_x_c += 1
                piece_y_max = max(i, piece_y_max)
    if not all((piece_x_sum, piece_x_c)):
        return (0, 0, 0, 0)
    piece_x = piece_x_sum / piece_x_c
    piece_y = piece_y_max - piece_base_height_1_2
    for i in range(int(h / 3), int(h * 2 / 3)):
        last_pixel = im_pixel[0, i]
        if board_x or board_y:
            break
        board_x_sum = 0
        board_x_c = 0
        for j in range(w):
            pixel = im_pixel[j, i]
            if abs(j - piece_x) < piece_body_width:
                continue
            if abs(pixel[0] - last_pixel[0]) + abs(pixel[1] - last_pixel[1]) + abs(pixel[2] - last_pixel[2]) > 10:
                board_x_sum += j
                board_x_c += 1
        if board_x_sum:
            board_x = board_x_sum / board_x_c
    board_y = piece_y - abs(board_x - piece_x) * math.sqrt(3) / 3
    if not all((board_x, board_y)):
        return (0, 0, 0, 0)
    return (piece_x, piece_y, board_x, board_y)

def main():
    if False:
        while True:
            i = 10
    while True:
        pull_screenshot()
        im = Image.open('./1.png')
        (piece_x, piece_y, board_x, board_y) = find_piece_and_board(im)
        ts = int(time.time())
        print(ts, piece_x, piece_y, board_x, board_y)
        if piece_x == 0:
            return
        set_button_position(im)
        distance = math.sqrt((board_x - piece_x) ** 2 + (board_y - piece_y) ** 2)
        jump(distance)
        save_debug_creenshot(ts, im, piece_x, piece_y, board_x, board_y)
        backup_screenshot(ts)
        time.sleep(random.uniform(1, 1.1))
if __name__ == '__main__':
    main()