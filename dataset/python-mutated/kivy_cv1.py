"""
参考：https://github.com/kivy/kivy/blob/master/kivy/core/camera/camera_opencv.py

kivy_cv1.py:
https://gist.github.com/ExpandOcean/de261e66949009f44ad2

pip install kivy

问题：无显示
"""
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2

class KivyCamera(Image):

    def __init__(self, capture, fps, **kwargs):
        if False:
            print('Hello World!')
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        if False:
            while True:
                i = 10
        (ret, frame) = self.capture.read()
        if ret:
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.texture = image_texture

class CamApp(App):

    def build(self):
        if False:
            i = 10
            return i + 15
        self.capture = cv2.VideoCapture(1)
        self.my_camera = KivyCamera(capture=self.capture, fps=30)
        return self.my_camera

    def on_stop(self):
        if False:
            return 10
        self.capture.release()
if __name__ == '__main__':
    CamApp().run()