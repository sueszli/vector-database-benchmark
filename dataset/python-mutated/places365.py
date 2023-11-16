import os
import numpy as np
import torch
from django.conf import settings
from PIL import Image
from torch.autograd import Variable as V
from torch.nn import functional as F
from torchvision import transforms as trn
import api.places365.wideresnet as wideresnet
from api.util import logger
torch.nn.Module.dump_patches = True
dir_places365_model = settings.PLACES365_ROOT

class Places365:
    labels_and_model_are_load = False

    def unload(self):
        if False:
            print('Hello World!')
        self.model = None
        self.classes = None
        self.W_attribute = None
        self.labels_IO = None
        self.labels_attribute = None
        self.labels_and_model_are_load = False

    def load(self):
        if False:
            while True:
                i = 10
        self.load_model()
        self.load_labels()
        self.labels_and_model_are_load = True

    def load_model(self):
        if False:
            for i in range(10):
                print('nop')

        def hook_feature(module, input, output):
            if False:
                i = 10
                return i + 15
            self.features_blobs.append(np.squeeze(output.data.cpu().numpy()))
        model_file = os.path.join(dir_places365_model, 'wideresnet18_places365.pth.tar')
        self.model = wideresnet.resnet18(num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, 'module.', ''): v for (k, v) in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        features_names = ['layer4', 'avgpool']
        for name in features_names:
            self.model._modules.get(name).register_forward_hook(hook_feature)

    def load_labels(self):
        if False:
            i = 10
            return i + 15
        file_path_category = os.path.join(dir_places365_model, 'categories_places365.txt')
        self.classes = list()
        with open(file_path_category) as class_file:
            for line in class_file:
                self.classes.append(line.strip().split(' ')[0][3:])
        self.classes = tuple(self.classes)
        file_path_IO = os.path.join(dir_places365_model, 'IO_places365.txt')
        with open(file_path_IO) as f:
            lines = f.readlines()
            self.labels_IO = []
            for line in lines:
                items = line.rstrip().split()
                self.labels_IO.append(int(items[-1]) - 1)
        self.labels_IO = np.array(self.labels_IO)
        file_path_attribute = os.path.join(dir_places365_model, 'labels_sunattribute.txt')
        with open(file_path_attribute) as f:
            lines = f.readlines()
            self.labels_attribute = [item.rstrip() for item in lines]
        file_path_W = os.path.join(dir_places365_model, 'W_sceneattribute_wideresnet18.npy')
        self.W_attribute = np.load(file_path_W)
        self.labels_are_load = True

    def returnTF(self):
        if False:
            return 10
        tf = trn.Compose([trn.Resize((224, 224)), trn.ToTensor(), trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return tf

    def remove_nonspace_separators(self, text):
        if False:
            for i in range(10):
                print('nop')
        return ' '.join(' '.join(' '.join(text.split('_')).split('/')).split('-'))

    def inference_places365(self, img_path, confidence):
        if False:
            return 10
        "\n        @param img_path: path to the image to generate labels from\n        @param confidence: minimum confidence before an category is selected\n        @return: {'environment': 'indoor'/'outdoor', 'categories': [...], 'attributes': [...]}\n        "
        try:
            if not self.labels_and_model_are_load:
                self.load()
            self.features_blobs = []
            tf = self.returnTF()
            params = list(self.model.parameters())
            weight_softmax = params[-2].data.numpy()
            weight_softmax[weight_softmax < 0] = 0
            img = Image.open(img_path)
            input_img = V(tf(img).unsqueeze(0))
            logit = self.model.forward(input_img)
            h_x = F.softmax(logit, 1).data.squeeze()
            (probs, idx) = h_x.sort(0, True)
            probs = probs.numpy()
            idx = idx.numpy()
            res = {}
            io_image = np.mean(self.labels_IO[idx[:10]])
            if io_image < 0.5:
                res['environment'] = 'indoor'
            else:
                res['environment'] = 'outdoor'
            res['categories'] = []
            for i in range(0, 5):
                if probs[i] > confidence:
                    res['categories'].append(self.remove_nonspace_separators(self.classes[idx[i]]))
                else:
                    break
            responses_attribute = self.W_attribute.dot(self.features_blobs[1])
            idx_a = np.argsort(responses_attribute)
            res['attributes'] = []
            for i in range(-1, -10, -1):
                res['attributes'].append(self.remove_nonspace_separators(self.labels_attribute[idx_a[i]]))
            return res
        except Exception as e:
            logger.error('Error in Places365 inference')
            raise e
place365_instance = Places365()