import warnings
import cv2
import dlib
from ..core import FaceDetector
from ...utils import load_file_from_url

class DlibDetector(FaceDetector):

    def __init__(self, device, path_to_detector=None, verbose=False):
        if False:
            return 10
        super().__init__(device, verbose)
        warnings.warn('Warning: this detector is deprecated. Please use a different one, i.e.: S3FD.')
        if 'cuda' in device:
            if path_to_detector is None:
                path_to_detector = load_file_from_url('https://www.adrianbulat.com/downloads/dlib/mmod_human_face_detector.dat')
            self.face_detector = dlib.cnn_face_detection_model_v1(path_to_detector)
        else:
            self.face_detector = dlib.get_frontal_face_detector()

    def detect_from_image(self, tensor_or_path):
        if False:
            for i in range(10):
                print('nop')
        image = self.tensor_or_path_to_ndarray(tensor_or_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        detected_faces = self.face_detector(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        if 'cuda' not in self.device:
            detected_faces = [[d.left(), d.top(), d.right(), d.bottom()] for d in detected_faces]
        else:
            detected_faces = [[d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()] for d in detected_faces]
        return detected_faces

    @property
    def reference_scale(self):
        if False:
            i = 10
            return i + 15
        return 195

    @property
    def reference_x_shift(self):
        if False:
            i = 10
            return i + 15
        return 0

    @property
    def reference_y_shift(self):
        if False:
            i = 10
            return i + 15
        return 0