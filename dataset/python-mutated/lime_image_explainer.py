from skimage.segmentation import mark_boundaries
from torchvision import transforms
import torch
from .lime import lime_image
import numpy as np
from .. import imagenet_utils, pytorch_utils, utils

class LimeImageExplainer:

    def __init__(self, model, predict_fn):
        if False:
            while True:
                i = 10
        self.model = model
        self.predict_fn = predict_fn

    def preprocess_input(self, inp):
        if False:
            return 10
        return inp

    def preprocess_label(self, label):
        if False:
            for i in range(10):
                print('nop')
        return label

    def explain(self, inp, ind=None, raw_inp=None, top_labels=5, hide_color=0, num_samples=1000, positive_only=True, num_features=5, hide_rest=True, pixel_val_max=255.0):
        if False:
            return 10
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(self.preprocess_input(raw_inp), self.predict_fn, top_labels=5, hide_color=0, num_samples=1000)
        (temp, mask) = explanation.get_image_and_mask(self.preprocess_label(ind) or explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
        img = mark_boundaries(temp / pixel_val_max, mask)
        img = torch.from_numpy(img)
        img = torch.transpose(img, 0, 2)
        img = torch.transpose(img, 1, 2)
        return img.unsqueeze(0)

class LimeImagenetExplainer(LimeImageExplainer):

    def __init__(self, model, predict_fn=None):
        if False:
            return 10
        super(LimeImagenetExplainer, self).__init__(model, predict_fn or self._imagenet_predict)

    def _preprocess_transform(self):
        if False:
            print('Hello World!')
        transf = transforms.Compose([transforms.ToTensor(), imagenet_utils.get_normalize_transform()])
        return transf

    def preprocess_input(self, inp):
        if False:
            i = 10
            return i + 15
        return np.array(imagenet_utils.get_resize_transform()(inp))

    def preprocess_label(self, label):
        if False:
            print('Hello World!')
        return label.item() if label is not None and utils.has_method(label, 'item') else label

    def _imagenet_predict(self, images):
        if False:
            while True:
                i = 10
        probs = imagenet_utils.predict(self.model, images, image_transform=self._preprocess_transform())
        return pytorch_utils.tensor2numpy(probs)