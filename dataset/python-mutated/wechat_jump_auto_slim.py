__author__ = 'Erimus'
'\n这个是精简版本，只取x轴距离。\n可以适配任意屏幕。\n把磁盘读写截图改为内存读写。\n可以防止被ban(从抓包数据看没有返回Error)。\n'
import os
import sys
import subprocess
import time
import random
from PIL import Image, ImageDraw
from io import BytesIO
VERSION = '1.1.4'
screenshot_way = 2

def check_screenshot():
    if False:
        for i in range(10):
            print('nop')
    global screenshot_way
    if screenshot_way < 0:
        print('暂不支持当前设备')
        sys.exit()
    binary_screenshot = pull_screenshot()
    try:
        Image.open(BytesIO(binary_screenshot)).load()
        print('Capture Method: {}'.format(screenshot_way))
    except Exception:
        screenshot_way -= 1
        check_screenshot()

def pull_screenshot():
    if False:
        for i in range(10):
            print('nop')
    global screenshot_way
    if screenshot_way in [1, 2]:
        process = subprocess.Popen('adb shell screencap -p', shell=True, stdout=subprocess.PIPE)
        screenshot = process.stdout.read()
        if screenshot_way == 2:
            binary_screenshot = screenshot.replace(b'\r\n', b'\n')
        else:
            binary_screenshot = screenshot.replace(b'\r\r\n', b'\n')
        return binary_screenshot
    elif screenshot_way == 0:
        os.system('adb shell screencap -p /sdcard/autojump.png')
        os.system('adb pull /sdcard/autojump.png .')

def find_piece_and_board(im):
    if False:
        while True:
            i = 10
    (w, h) = im.size
    im_pixel = im.load()

    def find_piece(pixel):
        if False:
            i = 10
            return i + 15
        return 40 < pixel[0] < 65 and 40 < pixel[1] < 65 and (80 < pixel[2] < 105)
    (piece_found, piece_fx, piece_fy) = (0, 0, 0)
    scan_piece_unit = w // 40
    ny = (h + w) // 2
    while ny > (h - w) // 2 and (not piece_found):
        ny -= scan_piece_unit
        for nx in range(0, w, scan_piece_unit):
            pixel = im_pixel[nx, ny]
            if find_piece(pixel):
                (piece_fx, piece_fy) = (nx, ny)
                piece_found = True
                break
    print('%-12s %s,%s' % ('piece_fuzzy:', piece_fx, piece_fy))
    if not piece_fx:
        return (0, 0)
    (piece_x, piece_x_set) = (0, [])
    piece_width = w // 14
    piece_height = w // 5
    for ny in range(piece_fy + scan_piece_unit, piece_fy - piece_height, -4):
        for nx in range(max(piece_fx - piece_width, 0), min(piece_fx + piece_width, w)):
            pixel = im_pixel[nx, ny]
            if find_piece(pixel):
                piece_x_set.append(nx)
        if len(piece_x_set) > 10:
            piece_x = sum(piece_x_set) / len(piece_x_set)
            break
    print('%-12s %s' % ('p_exact_x:', piece_x))
    board_x = 0
    if piece_x < w / 2:
        (board_x_start, board_x_end) = (w // 2, w)
    else:
        (board_x_start, board_x_end) = (0, w // 2)
    board_x_set = []
    for by in range((h - w) // 2, (h + w) // 2, 4):
        bg_pixel = im_pixel[0, by]
        for bx in range(board_x_start, board_x_end):
            pixel = im_pixel[bx, by]
            if abs(bx - piece_x) < piece_width:
                continue
            if abs(pixel[0] - bg_pixel[0]) + abs(pixel[1] - bg_pixel[1]) + abs(pixel[2] - bg_pixel[2]) > 10:
                board_x_set.append(bx)
        if len(board_x_set) > 10:
            board_x = sum(board_x_set) / len(board_x_set)
            print('%-12s %s' % ('target_x:', board_x))
            break
    return (piece_x, board_x)

def set_button_position(im, gameover=0):
    if False:
        print('Hello World!')
    (w, h) = im.size
    if h // 16 > w // 9 + 2:
        uih = int(w / 9 * 16)
    else:
        uih = h
    left = int(w / 2)
    top = int((h - uih) / 2 + uih * 0.825)
    if gameover:
        return (left, top)
    left = random.randint(w // 4, w - 20)
    top = random.randint(h * 3 // 4, h - 20)
    return (left, top)

def jump(piece_x, board_x, im, swipe_x1, swipe_y1):
    if False:
        return 10
    distanceX = abs(board_x - piece_x)
    shortEdge = min(im.size)
    jumpPercent = distanceX / shortEdge
    jumpFullWidth = 1700
    press_time = round(jumpFullWidth * jumpPercent)
    press_time = 0 if not press_time else max(press_time, 200)
    print('%-12s %.2f%% (%s/%s) | Press: %sms' % ('Distance:', jumpPercent * 100, distanceX, shortEdge, press_time))
    cmd = 'adb shell input swipe {x1} {y1} {x2} {y2} {duration}'.format(x1=swipe_x1, y1=swipe_y1, x2=swipe_x1 + random.randint(-10, 10), y2=swipe_y1 + random.randint(-10, 10), duration=press_time)
    os.system(cmd)

def main():
    if False:
        while True:
            i = 10
    check_screenshot()
    count = 0
    while True:
        count += 1
        print('---\n%-12s %s (%s)' % ('Times:', count, int(time.time())))
        binary_screenshot = pull_screenshot()
        im = Image.open(BytesIO(binary_screenshot))
        (w, h) = im.size
        if w > h:
            im = im.rotate(-90, expand=True)
        (piece_x, board_x) = find_piece_and_board(im)
        gameover = 0 if all((piece_x, board_x)) else 1
        (swipe_x1, swipe_y1) = set_button_position(im, gameover=gameover)
        jump(piece_x, board_x, im, swipe_x1, swipe_y1)
        wait = random.random() ** 5 * 9 + 1
        print('---\nWait %.3f s...' % wait)
        time.sleep(wait)
        print('Continue!')
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        os.system('adb kill-server')
        print('bye')
        exit(0)