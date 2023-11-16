import cv2
import os
import sys
import time
import datetime
from threading import Timer
from time import sleep
from datetime import datetime, timedelta
cap = cv2.VideoCapture(0)
cap.set(4, 800)
cap.set(4, 600)
cap.set(5, 15)
data_path = '/home/pi/Desktop/pi_system/raspi/dataDir/'
jpg_path = '/home/pi/Desktop/pi_system/raspi/jpgDir/'

def main():
    if False:
        i = 10
        return i + 15
    print('-- start --')
    while cap.isOpened():
        (ret, frame) = cap.read()
        time.sleep(60)
        now = datetime.now()
        writeStr = now.strftime('%y%m%d%H%M%S')
        saveImg = data_path + writeStr
        png_save = cv2.imwrite(saveImg + '.png', frame)
        os.system('scp ' + saveImg + ' ubuntu@18.179.32.84:/home/ubuntu/Desktop/hatapic/dataDir/' + writeStr + '.png')
        os.remove(png_save)
        saveImg2 = jpg_path + writeStr
        jpg_save = cv2.imwrite(saveImg2 + '.jpg', frame)
        os.system('scp ' + saveImg2 + ' ubuntu@18.179.32.84:/home/ubuntu/Desktop/hatapic/jpgDir/' + writeStr + '.jpg')
        os.remove(jpg_save)
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()