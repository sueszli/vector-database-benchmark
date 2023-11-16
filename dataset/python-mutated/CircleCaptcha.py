import io
import math
import operator
import sys
import urllib.request
from PIL import Image, ImageDraw
from ..base.ocr import BaseOCR

class ImageSequence:

    def __init__(self, im):
        if False:
            i = 10
            return i + 15
        self.im = im

    def __getitem__(self, ix):
        if False:
            for i in range(10):
                print('nop')
        try:
            if ix:
                self.im.seek(ix)
            return self.im
        except EOFError:
            raise IndexError

class CircleCaptcha(BaseOCR):
    __name__ = 'CircleCaptcha'
    __type__ = 'ocr'
    __version__ = '1.11'
    __status__ = 'testing'
    __description__ = 'Circle captcha ocr plugin'
    __license__ = 'GPLv3'
    __authors__ = [('Sasch', 'gsasch@gmail.com')]
    _DEBUG = False
    pointsofcirclefound = []
    BACKGROUND = 250
    BLACKCOLOR = 5

    def clean_image(self, im, pix):
        if False:
            print('Hello World!')
        cleandeep = 1
        imageheight = list(range(1, int(im.size[1])))
        imagewidth = list(range(1, int(im.size[0])))
        howmany = 0
        for y in imageheight:
            howmany = 0
            for x in imagewidth:
                curpix = pix[x, y]
                if curpix > self.BACKGROUND:
                    if howmany <= cleandeep and howmany > 0:
                        for ic in range(1, cleandeep + 1):
                            if x - ic > 0:
                                pix[x - ic, y] = self.BACKGROUND
                    howmany = 0
                elif howmany == 0:
                    howmany = howmany + 1
                else:
                    howmany = howmany + 1
            if howmany == 1:
                pix[x - 1, y] = self.BACKGROUND
        for x in imagewidth:
            howmany = 0
            for y in imageheight:
                curpix = pix[x, y]
                if curpix > self.BACKGROUND:
                    if howmany <= cleandeep and howmany > 0:
                        for ic in range(1, cleandeep + 1):
                            if y - ic > 0:
                                pix[x, y - ic] = self.BACKGROUND
                    howmany = 0
                elif howmany == 0:
                    howmany = howmany + 1
                else:
                    howmany = howmany + 1
            if howmany == 1:
                pix[x - 1, y] = self.BACKGROUND

    def find_first_pixel_x(self, im, pix, curx, cury, color=-1, ExitWithBlack=False):
        if False:
            while True:
                i = 10
        imagewidth = list(range(curx + 1, int(im.size[0])))
        jump = True
        newx = (-1, -1)
        blackfound = 0
        for x in imagewidth:
            curpix = pix[x, cury]
            if curpix < self.BLACKCOLOR:
                blackfound = blackfound + 1
                if ExitWithBlack is True and blackfound >= 3:
                    break
                else:
                    continue
            if curpix >= self.BACKGROUND:
                jump = False
                continue
            if curpix < self.BACKGROUND and color == -1 or (curpix == color and color > -1):
                if jump is False:
                    curcolor = curpix
                    newx = (x, curcolor)
                    break
        return newx

    def find_last_pixel_x(self, im, pix, curx, cury, color=-1, ExitWithBlack=False):
        if False:
            i = 10
            return i + 15
        imagewidth = list(range(curx + 1, int(im.size[0])))
        newx = (-1, -1)
        blackfound = 0
        for x in imagewidth:
            curpix = pix[x, cury]
            if curpix < self.BLACKCOLOR:
                blackfound = blackfound + 1
                if ExitWithBlack is True and blackfound >= 3:
                    break
                else:
                    continue
            if curpix >= self.BACKGROUND:
                if newx != (-1, -1):
                    break
            if curpix < self.BACKGROUND and color == -1 or (curpix == color and color > -1):
                curcolor = curpix
                newx = (x, curcolor)
        return newx

    def find_last_pixel_y(self, im, pix, curx, cury, DownToUp, color=-1, ExitWithBlack=False):
        if False:
            for i in range(10):
                print('nop')
        if DownToUp is False:
            imageheight = list(range(int(cury) + 1, int(im.size[1]) - 1))
        else:
            imageheight = list(range(int(cury) - 1, 1, -1))
        newy = (-1, -1)
        blackfound = 0
        for y in imageheight:
            curpix = pix[curx, y]
            if curpix < self.BLACKCOLOR:
                blackfound = blackfound + 1
                if ExitWithBlack is True and blackfound >= 3:
                    break
                else:
                    continue
            if curpix >= self.BACKGROUND:
                if newy != (-1, -1):
                    break
            if curpix < self.BACKGROUND and color == -1 or (curpix == color and color > -1):
                newy = (y, color)
        return newy

    def find_circle(self, pix, x1, y1, x2, y2, x3, y3):
        if False:
            while True:
                i = 10
        p2 = (x2 - x1, y2 - y1)
        p3 = (x3 - x1, y3 - y1)
        c = 0
        a = -1 * (p2[0] ** 2 - c) / p2[0]
        b = -1 * (p3[0] ** 2 + p3[1] ** 2 + a * p3[0] + c) / p3[1]
        r = math.floor(math.sqrt((-1 * (a / 2)) ** 2 + (-1 * (b / 2)) ** 2))
        cx = math.floor(-1 * (a / 2) + x1)
        cy = math.floor(-1 * (b / 2) + y1)
        return (cx, cy, r)

    def verify_circle_new(self, im, pix, c):
        if False:
            i = 10
            return i + 15
        '\n        This is the MAIN function to recognize the circle returns: 1 -> Found closed\n        circle 0 -> Found open circle.\n\n        -1 -> Not found circle\n        -2 -> Found black position then leave position\n        '
        imagewidth = list(range(int(c[0] - c[2]), int(c[0] + c[2])))
        min_ray = 15
        max_ray = 30
        exactfind = False
        howmany = 0
        missing = 0
        missinglist = []
        pointsofcircle = []
        if c[2] < min_ray or c[2] > max_ray:
            return -1
        if pix[c[0] + c[2], c[1]] < self.BLACKCOLOR:
            return -2
        if pix[c[0] - c[2], c[1]] < self.BLACKCOLOR:
            return -2
        if pix[c[0], c[1] + c[2]] < self.BLACKCOLOR:
            return -2
        if pix[c[0], c[1] - c[2]] < self.BLACKCOLOR:
            return -2
        cardinalpoints = 0
        if self.verify_point(im, pix, c[0] + c[2], c[1], True) == 1:
            cardinalpoints = cardinalpoints + 1
        if self.verify_point(im, pix, c[0] + c[2], c[1], False) == -1:
            return -2
        if self.verify_point(im, pix, c[0] - c[2], c[1], True) == 1:
            cardinalpoints = cardinalpoints + 1
        if self.verify_point(im, pix, c[0] - c[2], c[1], False) == -1:
            return -2
        if self.verify_point(im, pix, c[0], c[1] + c[2], True) == 1:
            cardinalpoints = cardinalpoints + 1
        if self.verify_point(im, pix, c[0], c[1] + c[2], False) == -1:
            return -2
        if self.verify_point(im, pix, c[0], c[1] - c[2], True) == 1:
            cardinalpoints = cardinalpoints + 1
        if self.verify_point(im, pix, c[0], c[1] - c[2], False) == -1:
            return -2
        if cardinalpoints < 3:
            return -1
        for x in imagewidth:
            y = int(round(c[1] - math.sqrt(c[2] ** 2 - (c[0] - x) ** 2)))
            y2 = int(round(c[1] + math.sqrt(c[2] ** 2 - (c[0] - x) ** 2)))
            howmany = howmany + 2
            if self.verify_point(im, pix, x, y, exactfind) == 0:
                missing = missing + 1
                missinglist.append((x, y))
            else:
                pointsofcircle.append((x, y))
            if self.verify_point(im, pix, x, y, False) == -1:
                return -2
            if self.verify_point(im, pix, x, y2, exactfind) == 0:
                missing = missing + 1
                missinglist.append((x, y2))
            else:
                pointsofcircle.append((x, y2))
            if self.verify_point(im, pix, x, y2, False) == -1:
                return -2

    def verify_circle(self, im, pix, c):
        if False:
            return 10
        '\n        This is the MAIN function to recognize the circle returns: 1 -> Found closed\n        circle 0 -> Found open circle.\n\n        -1 -> Not found circle\n        -2 -> Found black position then leave position\n        '
        imageheight = list(range(int(c[1] - c[2]), int(c[1] + c[2])))
        imagewidth = list(range(int(c[0] - c[2]), int(c[0] + c[2])))
        min_ray = 15
        max_ray = 30
        exactfind = False
        howmany = 0
        missing = 0
        missingconsecutive = 0
        missinglist = []
        minX = 0
        maxX = 0
        minY = 0
        maxY = 0
        pointsofcircle = []
        if c[2] < min_ray or c[2] > max_ray:
            return -1
        if pix[c[0] + c[2], c[1]] < self.BLACKCOLOR:
            return -2
        if pix[c[0] - c[2], c[1]] < self.BLACKCOLOR:
            return -2
        if pix[c[0], c[1] + c[2]] < self.BLACKCOLOR:
            return -2
        if pix[c[0], c[1] - c[2]] < self.BLACKCOLOR:
            return -2
        cardinalpoints = 0
        if self.verify_point(im, pix, c[0] + c[2], c[1], True) == 1:
            cardinalpoints = cardinalpoints + 1
        if self.verify_point(im, pix, c[0] + c[2], c[1], False) == -1:
            return -2
        if self.verify_point(im, pix, c[0] - c[2], c[1], True) == 1:
            cardinalpoints = cardinalpoints + 1
        if self.verify_point(im, pix, c[0] - c[2], c[1], False) == -1:
            return -2
        if self.verify_point(im, pix, c[0], c[1] + c[2], True) == 1:
            cardinalpoints = cardinalpoints + 1
        if self.verify_point(im, pix, c[0], c[1] + c[2], False) == -1:
            return -2
        if self.verify_point(im, pix, c[0], c[1] - c[2], True) == 1:
            cardinalpoints = cardinalpoints + 1
        if self.verify_point(im, pix, c[0], c[1] - c[2], False) == -1:
            return -2
        if cardinalpoints < 3:
            return -1
        for x in imagewidth:
            y = int(round(c[1] - math.sqrt(c[2] ** 2 - (c[0] - x) ** 2)))
            y2 = int(round(c[1] + math.sqrt(c[2] ** 2 - (c[0] - x) ** 2)))
            howmany = howmany + 2
            if self.verify_point(im, pix, x, y, exactfind) == 0:
                missing = missing + 1
                missinglist.append((x, y))
            else:
                pointsofcircle.append((x, y))
            if self.verify_point(im, pix, x, y, False) == -1:
                return -2
            if self.verify_point(im, pix, x, y2, exactfind) == 0:
                missing = missing + 1
                missinglist.append((x, y2))
            else:
                pointsofcircle.append((x, y2))
            if self.verify_point(im, pix, x, y2, False) == -1:
                return -2
        for y in imageheight:
            x = int(round(c[0] - math.sqrt(c[2] ** 2 - (c[1] - y) ** 2)))
            x2 = int(round(c[0] + math.sqrt(c[2] ** 2 - (c[1] - y) ** 2)))
            howmany = howmany + 2
            if self.verify_point(im, pix, x, y, exactfind) == 0:
                missing = missing + 1
                missinglist.append((x, y))
            else:
                pointsofcircle.append((x, y))
            if self.verify_point(im, pix, x, y, False) == -1:
                return -2
            if self.verify_point(im, pix, x2, y, exactfind) == 0:
                missing = missing + 1
                missinglist.append((x2, y))
            else:
                pointsofcircle.append((x2, y))
            if self.verify_point(im, pix, x2, y, exactfind) == -1:
                return -2
        for p in missinglist:
            if self.verify_point(im, pix, p[0] - 1, p[1], exactfind) == 1 and self.verify_point(im, pix, p[0], p[1] + 1, exactfind) == 1:
                missing = missing - 1
            elif self.verify_point(im, pix, p[0] - 1, p[1], exactfind) == 1 and self.verify_point(im, pix, p[0], p[1] - 1, exactfind) == 1:
                missing = missing - 1
            elif self.verify_point(im, pix, p[0] + 1, p[1], exactfind) == 1 and self.verify_point(im, pix, p[0], p[1] + 1, exactfind) == 1:
                missing = missing - 1
            elif self.verify_point(im, pix, p[0] + 1, p[1], exactfind) == 1 and self.verify_point(im, pix, p[0], p[1] - 1, exactfind) == 1:
                missing = missing - 1
            if (p[0], p[1] + 1) in missinglist or (p[0], p[1] - 1) in missinglist or (p[0] + 1, p[1]) in missinglist or ((p[0] - 1, p[1]) in missinglist) or ((p[0] + 1, p[1] + 1) in missinglist) or ((p[0] - 1, p[1] + 1) in missinglist) or ((p[0] + 1, p[1] - 1) in missinglist) or ((p[0] - 1, p[1] - 1) in missinglist) or (self.verify_point(im, pix, p[0], p[1], False) == 1):
                missingconsecutive = missingconsecutive + 1
        if len(missinglist) > 0:
            minX = min(missinglist, key=operator.itemgetter(0))[0]
            maxX = max(missinglist, key=operator.itemgetter(0))[0]
            minY = min(missinglist, key=operator.itemgetter(1))[1]
            maxY = max(missinglist, key=operator.itemgetter(1))[1]
        if self.pyload.debug:
            self.log_debug('Center: {}'.format(c), 'Missing: {}'.format(missing), 'Howmany: {}'.format(howmany), 'Ratio: {}'.format(missing / howmany), 'Missing consecutives: {}'.format(missingconsecutive), 'Missing X lenght: {}:{}'.format(minX, maxX), 'Missing Y lenght: {}:{}'.format(minY, maxY), 'Ratio without consecutives: {}'.format((missing - missingconsecutive) / howmany), 'List missing: {}'.format(missinglist))
        if maxX - minX >= c[2] * 2 * 0.75:
            return -1
        if maxY - minY >= c[2] * 2 * 0.75:
            return -1
        '\n        #: Lenght of missing cannot be less 10% of diameter\n        if maxX - minX < c[2] * 2 * 0.10 and maxY - minY < c[2] * 2 * 0.10:\n            return -1\n        '
        if missing / howmany > 0.25 or missingconsecutive >= howmany / 4 * 2 or howmany < 80:
            return -1
        elif missing == 0:
            self.pointsofcirclefound.extend(pointsofcircle)
            return 1
        elif (missing - missingconsecutive) / howmany < 0.2:
            return 0
        else:
            self.pointsofcirclefound.extend(pointsofcircle)
            return 1

    def verify_point(self, im, pix, x, y, exact, color=-1):
        if False:
            return 10
        result = 0
        if x < 0 or x >= im.size[0]:
            return result
        if y < 0 or y >= im.size[1]:
            return result
        curpix = pix[x, y]
        if curpix == color and color > -1 or (curpix < self.BACKGROUND and color == -1):
            if curpix > self.BLACKCOLOR:
                result = 1
            else:
                result = -1
        if exact is False:
            if x + 1 < im.size[0]:
                curpix = pix[x + 1, y]
                if curpix == color and color > -1 or (curpix < self.BACKGROUND and color == -1):
                    if curpix > self.BLACKCOLOR:
                        result = 1
                if curpix <= self.BLACKCOLOR:
                    result = -1
            if x > 0:
                curpix = pix[x - 1, y]
                if curpix == color and color > -1 or (curpix < self.BACKGROUND and color == -1):
                    if curpix > self.BLACKCOLOR:
                        result = 1
                if curpix <= self.BLACKCOLOR:
                    result = -1
        return result

    def decrypt(self, img):
        if False:
            for i in range(10):
                print('nop')
        i_debug_save_file = 0
        mypalette = None
        for im in ImageSequence(img):
            im.save('orig.png', 'png')
            if mypalette is not None:
                im.putpalette(mypalette)
            mypalette = im.getpalette()
            im = im.convert('L')
            if self.pyload.debug:
                i_debug_save_file = i_debug_save_file + 1
                im.save('output{}.png'.format(i_debug_save_file), 'png')
                input('frame: {}'.format(im))
            pix = im.load()
            stepheight = list(range(1, im.size[1], 2))
            lst_points = []
            lstX = []
            lstY = []
            lst_colors = []
            min_distance = 10
            max_diameter = 70
            if self.pyload.debug:
                imdebug = im.copy()
                draw = ImageDraw.Draw(imdebug)
                pixcopy = imdebug.load()
            self.clean_image(im, pix)
            im.save('cleaned{}.png'.format(i_debug_save_file), 'png')
            found = set()
            findnewcircle = True
            for y1 in stepheight:
                x1 = 1
                for k in range(1, 100):
                    findnewcircle = False
                    retval = self.find_first_pixel_x(im, pix, x1, y1, -1, False)
                    x1 = retval[0]
                    if x1 == -2:
                        break
                    if x1 == -1:
                        break
                    if self.pyload.debug:
                        self.log_debug(f'x1, y1 -> {(x1, y1)}: {pix[x1, y1]}')
                    if (x1, y1) in self.pointsofcirclefound:
                        if self.pyload.debug:
                            self.log_debug(f'Found {(x1, y1)}')
                        continue
                    if self.pyload.debug:
                        pixcopy[x1, y1] = 45
                    x2 = x1
                    y2 = y1
                    for i in range(1, 100):
                        retval = self.find_last_pixel_x(im, pix, x2, y2, -1, True)
                        x2 = retval[0]
                        if x1 == -2:
                            findnewcircle = True
                            break
                        if x2 == -1:
                            break
                        if self.pyload.debug:
                            self.log_debug('x2, y2 -> {}: {}'.format((x2, y1), pix[x2, y1]))
                        if abs(x2 - x1) < min_distance:
                            continue
                        if abs(x2 - x1) > im.size[1] * 2 / 3:
                            break
                        if abs(x2 - x1) > max_diameter:
                            break
                        if self.pyload.debug:
                            pixcopy[x2, y2] = 65
                        for invert in range(2):
                            x3 = math.floor(x2 - (x2 - x1) / 2)
                            y3 = y1
                            for j in range(1, 50):
                                retval = self.find_last_pixel_y(im, pix, x3, y3, invert == 1, -1, True)
                                y3 = retval[0]
                                if y3 == -2:
                                    findnewcircle = True
                                    break
                                if y3 == -1:
                                    break
                                if self.pyload.debug:
                                    self.log_debug('x3, y3 -> ' + str((x3, y3)) + ': ' + str(pix[x3, y3]))
                                if abs(y3 - y2) < min_distance:
                                    continue
                                if abs(y3 - y2) > im.size[1] * 2 / 3:
                                    break
                                if abs(y3 - y2) > max_diameter:
                                    break
                                if self.pyload.debug:
                                    pixcopy[x3, y3] = 85
                                c = self.find_circle(pix, x1, y1, x2, y2, x3, y3)
                                if c[0] + c[2] >= im.size[0] or c[1] + c[2] >= im.size[1] or c[0] - c[2] <= 0 or (c[1] - c[2] <= 0):
                                    continue
                                if self.pyload.debug:
                                    pixcopy[c[0], c[1]] = 0
                                verified = self.verify_circle(im, pix, c)
                                if verified == -1:
                                    verified = -1
                                elif verified == 0:
                                    found.add(((c[0], c[1], c[2]), verified))
                                    findnewcircle = True
                                elif verified == 1:
                                    found.add(((c[0], c[1], c[2]), verified))
                                    findnewcircle = True
                                if self.pyload.debug:
                                    _pause = ''
                                    if verified == 0:
                                        draw.ellipse((c[0] - c[2], c[1] - c[2], c[0] + c[2], c[1] + c[2]), outline=120)
                                        _pause = 'OPENED'
                                    if verified == 1:
                                        draw.ellipse((c[0] - c[2], c[1] - c[2], c[0] + c[2], c[1] + c[2]), outline=65)
                                        _pause = 'CLOSED'
                                    imdebug.save('debug.png', 'png')
                                    if _pause != '':
                                        valore = input('Found ' + _pause + ' CIRCLE circle press [Enter] = continue / [q] for Quit: ' + str(verified))
                                        if valore == 'q':
                                            sys.exit()
                                if findnewcircle is True:
                                    break
                            if findnewcircle is True:
                                break
                        if findnewcircle is True:
                            break
            if self.pyload.debug:
                self.log_debug('Howmany opened circle?', found)
            for c in found:
                verify = c[1]
                if verify == 0:
                    p = c[0]
                    if ((p[0], p[1] + 1, p[2]), 1) in found or ((p[0], p[1] - 1, p[2]), 1) in found or ((p[0] + 1, p[1], p[2]), 1) in found or (((p[0] - 1, p[1], p[2]), 1) in found) or (((p[0] + 1, p[1] + 1, p[2]), 1) in found) or (((p[0] - 1, p[1] + 1, p[2]), 1) in found) or (((p[0] + 1, p[1] - 1, p[2]), 1) in found) or (((p[0] - 1, p[1] - 1, p[2]), 1) in found):
                        verify = -1
                    if ((p[0], p[1] + 1, p[2] + 1), 1) in found or ((p[0], p[1] - 1, p[2] + 1), 1) in found or ((p[0] + 1, p[1], p[2] + 1), 1) in found or (((p[0] - 1, p[1], p[2] + 1), 1) in found) or (((p[0] + 1, p[1] + 1, p[2] + 1), 1) in found) or (((p[0] - 1, p[1] + 1, p[2] + 1), 1) in found) or (((p[0] + 1, p[1] - 1, p[2] + 1), 1) in found) or (((p[0] - 1, p[1] - 1, p[2] + 1), 1) in found):
                        verify = -1
                    if ((p[0], p[1] + 1, p[2] - 1), 1) in found or ((p[0], p[1] - 1, p[2] - 1), 1) in found or ((p[0] + 1, p[1], p[2] - 1), 1) in found or (((p[0] - 1, p[1], p[2] - 1), 1) in found) or (((p[0] + 1, p[1] + 1, p[2] - 1), 1) in found) or (((p[0] - 1, p[1] + 1, p[2] - 1), 1) in found) or (((p[0] + 1, p[1] - 1, p[2] - 1), 1) in found) or (((p[0] - 1, p[1] - 1, p[2] - 1), 1) in found):
                        verify = -1
        if self.pyload.debug:
            im.save('output.png', 'png')

    def decrypt_from_web(self, url):
        if False:
            while True:
                i = 10
        file = io.StringIO(urllib.request.urlopen(url).read())
        img = Image.open(file)
        coords = self.decrypt(img)
        self.log_info(self._('Coords: {}').format(coords))

    def decrypt_from_file(self, filename):
        if False:
            return 10
        coords = self.decrypt(Image.open(filename))
        self.log_info(self._('Coords: {}').format(coords))