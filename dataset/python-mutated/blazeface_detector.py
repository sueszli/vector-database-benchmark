from torch.utils.model_zoo import load_url
from ..core import FaceDetector
from ...utils import load_file_from_url
from .net_blazeface import BlazeFace
from .detect import *
models_urls = {'blazeface_weights': 'https://github.com/hollance/BlazeFace-PyTorch/blob/master/blazeface.pth?raw=true', 'blazeface_anchors': 'https://github.com/hollance/BlazeFace-PyTorch/blob/master/anchors.npy?raw=true', 'blazeface_back_weights': 'https://github.com/hollance/BlazeFace-PyTorch/blob/master/blazefaceback.pth?raw=true', 'blazeface_back_anchors': 'https://github.com/hollance/BlazeFace-PyTorch/blob/master/anchorsback.npy?raw=true'}

class BlazeFaceDetector(FaceDetector):

    def __init__(self, device, back_model=False, path_to_detector=None, path_to_anchor=None, verbose=False, min_score_thresh=0.5, min_suppression_threshold=0.3):
        if False:
            print('Hello World!')
        super(BlazeFaceDetector, self).__init__(device, verbose)
        self.back_model = back_model
        if path_to_detector is None:
            if back_model:
                model_weights = load_url(models_urls['blazeface_back_weights'])
                model_anchors = np.load(load_file_from_url(models_urls['blazeface_back_anchors']))
            else:
                model_weights = load_url(models_urls['blazeface_weights'])
                model_anchors = np.load(load_file_from_url(models_urls['blazeface_anchors']))
        else:
            model_weights = torch.load(path_to_detector)
            model_anchors = np.load(path_to_anchor)
        self.face_detector = BlazeFace(back_model=back_model)
        self.face_detector.load_state_dict(model_weights)
        self.face_detector.load_anchors_from_npy(model_anchors, device)
        self.face_detector.min_score_thresh = min_score_thresh
        self.face_detector.min_suppression_threshold = min_suppression_threshold
        self.face_detector.to(device)
        self.face_detector.eval()

    def detect_from_image(self, tensor_or_path):
        if False:
            i = 10
            return i + 15
        image = self.tensor_or_path_to_ndarray(tensor_or_path)
        image_size = 256 if self.back_model else 128
        bboxlist = detect(self.face_detector, image, target_size=image_size, device=self.device)[0]
        return bboxlist

    def detect_from_batch(self, tensor):
        if False:
            print('Hello World!')
        image_size = 256 if self.back_model else 128
        bboxlists = batch_detect(self.face_detector, tensor, target_size=image_size, device=self.device)
        return bboxlists

    @property
    def reference_scale(self):
        if False:
            return 10
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
            print('Hello World!')
        return 0