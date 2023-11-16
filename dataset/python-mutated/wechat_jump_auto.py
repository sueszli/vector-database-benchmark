"""
=== 思路 ===
核心：每次落稳之后截图，根据截图算出棋子的坐标和下一个块顶面的中点坐标，
    根据两个点的距离乘以一个时间系数获得长按的时间
识别棋子：靠棋子的颜色来识别位置，通过截图发现最下面一行大概是一条
    直线，就从上往下一行一行遍历，比较颜色（颜色用了一个区间来比较）
    找到最下面的那一行的所有点，然后求个中点，求好之后再让 Y 轴坐标
    减小棋子底盘的一半高度从而得到中心点的坐标
识别棋盘：靠底色和方块的色差来做，从分数之下的位置开始，一行一行扫描，
    由于圆形的块最顶上是一条线，方形的上面大概是一个点，所以就
    用类似识别棋子的做法多识别了几个点求中点，这时候得到了块中点的 X
    轴坐标，这时候假设现在棋子在当前块的中心，根据一个通过截图获取的
    固定的角度来推出中点的 Y 坐标
最后：根据两点的坐标算距离乘以系数来获取长按时间（似乎可以直接用 X 轴距离）
"""
import math
import re
import random
import sys
import time
from PIL import Image
from six.moves import input
if sys.version_info.major != 3:
    print('请使用Python3')
    exit(1)
try:
    from common import debug, config, screenshot, UnicodeStreamFilter
    from common.auto_adb import auto_adb
except Exception as ex:
    print(ex)
    print('请将脚本放在项目根目录中运行')
    print('请检查项目根目录中的 common 文件夹是否存在')
    exit(1)
adb = auto_adb()
VERSION = '1.1.4'
DEBUG_SWITCH = False
adb.test_device()
config = config.open_accordant_config()
under_game_score_y = config['under_game_score_y']
press_coefficient = config['press_coefficient']
piece_base_height_1_2 = config['piece_base_height_1_2']
piece_body_width = config['piece_body_width']
head_diameter = config.get('head_diameter')
if head_diameter == None:
    density_str = adb.test_density()
    matches = re.search('\\d+', density_str)
    density_val = int(matches.group(0))
    head_diameter = density_val / 8

def set_button_position(im):
    if False:
        i = 10
        return i + 15
    '\n    将 swipe 设置为 `再来一局` 按钮的位置\n    '
    global swipe_x1, swipe_y1, swipe_x2, swipe_y2
    (w, h) = im.size
    left = int(w / 2)
    top = int(1584 * (h / 1920.0))
    left = int(random.uniform(left - 200, left + 200))
    top = int(random.uniform(top - 200, top + 200))
    after_top = int(random.uniform(top - 200, top + 200))
    after_left = int(random.uniform(left - 200, left + 200))
    (swipe_x1, swipe_y1, swipe_x2, swipe_y2) = (left, top, after_left, after_top)

def jump(distance, delta_piece_y):
    if False:
        i = 10
        return i + 15
    '\n    跳跃一定的距离\n    '
    scale = 0.945 * 2 / head_diameter
    actual_distance = distance * scale * (math.sqrt(6) / 2)
    press_time = (-945 + math.sqrt(945 ** 2 + 4 * 105 * 36 * actual_distance)) / (2 * 105) * 1000
    press_time *= press_coefficient
    press_time = max(press_time, 200)
    press_time = int(press_time)
    cmd = 'shell input swipe {x1} {y1} {x2} {y2} {duration}'.format(x1=swipe_x1, y1=swipe_y1, x2=swipe_x2, y2=swipe_y2, duration=press_time + delta_piece_y)
    print(cmd)
    adb.run(cmd)
    return press_time

def find_piece_and_board(im):
    if False:
        print('Hello World!')
    '\n    寻找关键坐标\n    '
    (w, h) = im.size
    points = []
    piece_y_max = 0
    board_x = 0
    board_y = 0
    scan_x_border = int(w / 8)
    scan_start_y = 0
    im_pixel = im.load()
    for i in range(int(h / 3), int(h * 2 / 3), 50):
        last_pixel = im_pixel[0, i]
        for j in range(1, w):
            pixel = im_pixel[j, i]
            if pixel != last_pixel:
                scan_start_y = i - 50
                break
        if scan_start_y:
            break
    print('start scan Y axis: {}'.format(scan_start_y))
    for i in range(scan_start_y, int(h * 2 / 3)):
        for j in range(scan_x_border, w - scan_x_border):
            pixel = im_pixel[j, i]
            if 50 < pixel[0] < 60 and 53 < pixel[1] < 63 and (95 < pixel[2] < 110):
                points.append((j, i))
                piece_y_max = max(i, piece_y_max)
    bottom_x = [x for (x, y) in points if y == piece_y_max]
    if not bottom_x:
        return (0, 0, 0, 0, 0)
    piece_x = int(sum(bottom_x) / len(bottom_x))
    piece_y = piece_y_max - piece_base_height_1_2
    if piece_x < w / 2:
        board_x_start = piece_x
        board_x_end = w
    else:
        board_x_start = 0
        board_x_end = piece_x
    for i in range(int(h / 3), int(h * 2 / 3)):
        last_pixel = im_pixel[0, i]
        if board_x or board_y:
            break
        board_x_sum = 0
        board_x_c = 0
        for j in range(int(board_x_start), int(board_x_end)):
            pixel = im_pixel[j, i]
            if abs(j - piece_x) < piece_body_width:
                continue
            ver_pixel = im_pixel[j, i + 5]
            if abs(pixel[0] - last_pixel[0]) + abs(pixel[1] - last_pixel[1]) + abs(pixel[2] - last_pixel[2]) > 10 and abs(ver_pixel[0] - last_pixel[0]) + abs(ver_pixel[1] - last_pixel[1]) + abs(ver_pixel[2] - last_pixel[2]) > 10:
                board_x_sum += j
                board_x_c += 1
        if board_x_sum:
            board_x = board_x_sum / board_x_c
    last_pixel = im_pixel[board_x, i]
    center_x = w / 2 + 24 / 1080 * w
    center_y = h / 2 + 17 / 1920 * h
    if piece_x > center_x:
        board_y = round(25.5 / 43.5 * (board_x - center_x) + center_y)
        delta_piece_y = piece_y - round(25.5 / 43.5 * (piece_x - center_x) + center_y)
    else:
        board_y = round(-(25.5 / 43.5) * (board_x - center_x) + center_y)
        delta_piece_y = piece_y - round(-(25.5 / 43.5) * (piece_x - center_x) + center_y)
    if not all((board_x, board_y)):
        return (0, 0, 0, 0, 0)
    return (piece_x, piece_y, board_x, board_y, delta_piece_y)

def yes_or_no():
    if False:
        return 10
    '\n    检查是否已经为启动程序做好了准备\n    '
    while True:
        yes_or_no = str(input('请确保手机打开了 ADB 并连接了电脑，然后打开跳一跳并【开始游戏】后再用本程序，确定开始？[y/n]:'))
        if yes_or_no == 'y':
            break
        elif yes_or_no == 'n':
            print('谢谢使用', end='')
            exit(0)
        else:
            print('请重新输入')

def main():
    if False:
        return 10
    '\n    主函数\n    '
    print('程序版本号：{}'.format(VERSION))
    print('激活窗口并按 CONTROL + C 组合键退出')
    debug.dump_device_info()
    screenshot.check_screenshot()
    (i, next_rest, next_rest_time) = (0, random.randrange(3, 10), random.randrange(5, 10))
    while True:
        im = screenshot.pull_screenshot()
        (piece_x, piece_y, board_x, board_y, delta_piece_y) = find_piece_and_board(im)
        ts = int(time.time())
        print(ts, piece_x, piece_y, board_x, board_y)
        set_button_position(im)
        jump(math.sqrt((board_x - piece_x) ** 2 + (board_y - piece_y) ** 2), delta_piece_y)
        if DEBUG_SWITCH:
            debug.save_debug_screenshot(ts, im, piece_x, piece_y, board_x, board_y)
            debug.backup_screenshot(ts)
        im.close()
        i += 1
        if i == next_rest:
            print('已经连续打了 {} 下，休息 {}秒'.format(i, next_rest_time))
            for j in range(next_rest_time):
                sys.stdout.write('\r程序将在 {}秒 后继续'.format(next_rest_time - j))
                sys.stdout.flush()
                time.sleep(1)
            print('\n继续')
            (i, next_rest, next_rest_time) = (0, random.randrange(30, 100), random.randrange(10, 60))
        time.sleep(random.uniform(1.2, 1.4))
if __name__ == '__main__':
    try:
        yes_or_no()
        main()
    except KeyboardInterrupt:
        adb.run('kill-server')
        print('\n谢谢使用', end='')
        exit(0)