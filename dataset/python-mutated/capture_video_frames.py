import os
import shutil
import sys
import cv2

class FrameCapture:
    """
        Class definition to capture frames
    """

    def __init__(self, file_path):
        if False:
            return 10
        '\n            initializing directory where the captured frames will be stored.\n            Also truncating the directory where captured frames are stored, if exists.\n        '
        self.directory = 'captured_frames'
        self.file_path = file_path
        if os.path.exists(self.directory):
            shutil.rmtree(self.directory)
        os.mkdir(self.directory)

    def capture_frames(self):
        if False:
            i = 10
            return i + 15
        '\n            This method captures the frames from the video file provided.\n            This program makes use of openCV library\n        '
        cv2_object = cv2.VideoCapture(self.file_path)
        frame_number = 0
        frame_found = 1
        while frame_found:
            (frame_found, image) = cv2_object.read()
            capture = f'{self.directory}/frame{frame_number}.jpg'
            cv2.imwrite(capture, image)
            frame_number += 1
if __name__ == '__main__':
    file_path = sys.argv[1]
    fc = FrameCapture(file_path)
    fc.capture_frames()