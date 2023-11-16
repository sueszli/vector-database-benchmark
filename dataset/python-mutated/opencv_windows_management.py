"""
opencv_windows_management.py:
"""
import cv2, math
import tkinter as tk

class Window:

    def __init__(self, name, image, weight=1):
        if False:
            return 10
        self.name = name
        self.image = image.copy()
        self.weight = weight
        self.shape = self.image.shape
        self.hight_x = self.shape[0]
        self.lenght_y = self.shape[1]

class opencv_windows_management:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.windows = dict()
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        self.screen_size = (screen_width, screen_height)
        root.quit()

    def add(self, name, image, weight=1):
        if False:
            return 10
        '\n        权重,越高，图片显示越大\n        :return:\n        '
        cv2.namedWindow(name, flags=cv2.WINDOW_AUTOSIZE)
        window = Window(name, image, weight)
        self.windows[name] = window

    def show(self):
        if False:
            return 10
        lenw = len(self.windows)
        w_l = int(self.screen_size[0] / lenw)
        max_num_line = math.ceil(math.sqrt(lenw))
        for (i, name) in enumerate(self.windows):
            win = self.windows[name]
            image = win.image
            h_x = int(w_l / win.lenght_y * win.hight_x)
            img2 = cv2.resize(image, (w_l, h_x))
            cv2.moveWindow(name, w_l * i, 0)
            cv2.imshow(name, img2)