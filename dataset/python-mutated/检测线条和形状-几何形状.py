"""
检测线条和形状-几何形状.py:

https://stackoverflow.com/questions/31974843/detecting-lines-and-shapes-in-opencv-using-python

"""
import cv2
import numpy as np

class File(object):

    def __init__(self, filename):
        if False:
            return 10
        self.filename = filename

    def open(self, filename=None, mode='r'):
        if False:
            for i in range(10):
                print('nop')
        if filename is None:
            filename = self.filename
        return (cv2.imread(filename), open(filename, mode))

    def save(self, image=None, filename_override=None):
        if False:
            return 10
        filename = 'output/' + self.filename.split('/')[-1]
        if filename_override:
            filename = 'output/' + filename_override
        return cv2.imwrite(filename, image)

class Image(object):

    def __init__(self, image):
        if False:
            for i in range(10):
                print('nop')
        self.image = image

    def grayscale(self):
        if False:
            return 10
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def edges(self):
        if False:
            while True:
                i = 10
        return cv2.Canny(self.image, 0, 255)

    def lines(self):
        if False:
            return 10
        lines = cv2.HoughLinesP(self.image, 1, np.pi / 2, 6, None, 50, 10)
        for line in lines[0]:
            pt1 = (line[0], line[1])
            pt2 = (line[2], line[3])
            cv2.line(self.image, pt1, pt2, (0, 0, 255), 2)
if __name__ == '__main__':
    File = File('images/a.png')
    Image = Image(File.open()[0])
    Image.image = Image.grayscale()
    Image.lines()
    File.save(Image.image)