from cvs import *
import facerecognition
import numpy as np

class MyApp(App):

    def __init__(self, *args):
        if False:
            return 10
        super(MyApp, self).__init__(*args)

    def main(self):
        if False:
            return 10
        main_container = gui.VBox(width=360, height=680, style={'margin': '0px auto'})
        self.aidcam = OpencvVideoWidget(self, width=340, height=480)
        self.aidcam.style['margin'] = '10px'
        self.aidcam.set_identifier('myimage_receiver')
        main_container.append(self.aidcam)
        self.txt = gui.TextInput(width=200, height=30, margin='10px')
        self.txt.set_text('usename')
        self.txt.onchange.do(self.on_text_area_change)
        self.bt = gui.Button('Add Person!', width=200, height=30, margin='10px')
        self.bt.onclick.do(self.on_button_pressed)
        main_container.append(self.txt)
        main_container.append(self.bt)
        return main_container

    def on_text_area_change(self, widget, newValue):
        if False:
            i = 10
            return i + 15
        print('Text Area value changed!')

    def on_button_pressed(self, widget):
        if False:
            return 10
        userId = self.txt.get_text()
        cvs.setLbs(userId)
        self.bt.set_text('success!')

def process():
    if False:
        for i in range(10):
            print('nop')
    cap = cvs.VideoCapture(1)
    facerecog = facerecognition.FaceRecognition('./models', 0.73)
    while True:
        sleep(30)
        img = cap.read()
        if img is None:
            continue
        image_char = img.astype(np.uint8).tostring()
        userId = cvs.getLbs()
        if userId != '':
            ret = facerecog.add_person(userId, img.shape[0], img.shape[1], image_char)
            if ret == 0:
                print('you add_person is success!')
            else:
                print('you add_person is failed!')
            userId = ''
            cvs.setLbs(userId)
            continue
        rets = facerecog.recognize(img.shape[0], img.shape[1], image_char)
        for ret in rets:
            print('draw bounding box for the face')
            rect = ret['rect']
            p1 = (int(rect[0]), int(rect[1]))
            p2 = (int(rect[0] + rect[2]), int(rect[1] + rect[2]))
            cv2.rectangle(img, p1, p2, (0, 255, 0), 3, 1)
            cv2.putText(img, ret['name'], (int(rect[0]), int(rect[1]) - 30), cv2.FONT_ITALIC, 2, (77, 255, 9), 2)
        cvs.imshow(img)
if __name__ == '__main__':
    initcv(process)
    startcv(MyApp)