from cvs import *
import numpy as np

class MyApp(App):

    def __init__(self, *args):
        if False:
            return 10
        super(MyApp, self).__init__(*args)

    def main(self):
        if False:
            print('Hello World!')
        main_container = gui.VBox(width=360, height=680, style={'margin': '0px auto'})
        self.aidcam = OpencvVideoWidget(self, width=340, height=480)
        self.aidcam.style['margin'] = '10px'
        self.aidcam.set_identifier('myimage_receiver')
        main_container.append(self.aidcam)
        return main_container

def process():
    if False:
        print('Hello World!')
    cap = cvs.VideoCapture(1)
    while True:
        sleep(30)
        img = cap.read()
        if img is None:
            continue
        cvs.imshow(img)
if __name__ == '__main__':
    initcv(process)
    startcv(MyApp)