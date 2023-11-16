import reconmodels
from sklearn.metrics import confusion_matrix
from recon_data_generator import transform, clean_video_list, get_samples
from moviepy.editor import VideoFileClip
import numpy as np
import torch
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import re
import recon_data_generator
import recon_losses_R
import recon_metrics_R
import yaml
import time
import warnings
'\nCreated on Thu Oct 26 11:06:51 2017\n@author: Utku Ozbulak - github.com/utkuozbulak\n'
from PIL import Image
import numpy as np
import torch
from misc_functions import get_example_params, save_class_activation_images

class CamExtractor:
    """
        Extracts cam features from the model
    """

    def __init__(self, model, target_layer):
        if False:
            for i in range(10):
                print('nop')
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        if False:
            i = 10
            return i + 15
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        if False:
            while True:
                i = 10
        '\n            Does a forward pass on convolutions, hooks the function at given layer\n        '
        conv_output = None
        for (module_pos, module) in self.model.features._modules.items():
            x = module(x)
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x
        return (conv_output, x)

    def forward_pass(self, x):
        if False:
            for i in range(10):
                print('nop')
        '\n            Does a full forward pass on the model\n        '
        (conv_output, x) = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)
        x = self.model.classifier(x)
        return (conv_output, x)

class GradCam:
    """
        Produces class activation map
    """

    def __init__(self, model, target_layer):
        if False:
            while True:
                i = 10
        self.model = model
        self.model.eval()
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        if False:
            i = 10
            return i + 15
        (conv_output, model_output) = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        target = conv_output.data.numpy()[0]
        weights = np.mean(guided_gradients, axis=(1, 2))
        cam = np.ones(target.shape[1:], dtype=np.float32)
        for (i, w) in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        cam = np.uint8(cam * 255)
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2], input_image.shape[3]), Image.ANTIALIAS)) / 255
        return cam
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    torch.cuda.empty_cache()
    with open('config.yaml') as f:
        config = yaml.load(f)
    eval_frame_data_set = recon_data_generator.FrameDataSet(config, status='eval')
    eval_frame_train_loader = torch.utils.data.DataLoader(eval_frame_data_set, batch_size=1, shuffle=True, num_workers=0)
    print('number of samples in EVAL data set : ', eval_frame_data_set.__len__())
    image_model = reconmodels.resnet50withcbam()
    image_model.load_state_dict(torch.load(config['save_path'] + 'imgbest.pth'))
    image_model.cuda()
    image_model.eval()
    print('*' * 20)
    total_eval_sample = 0
    total_eval_correct = 0
    for (i, data) in enumerate(eval_frame_train_loader):
        with torch.no_grad():
            image = data[0].unsqueeze(1)
            label = data[1]
            image = torch.tensor(image).cuda()
            label = torch.tensor(label)
            torch.cuda.empty_cache()
            label = label.to(device=torch.cuda.current_device(), dtype=torch.long)
            grad_cam = GradCam(image_model, target_layer=4)
            out = image_model.forward(image)
            (_, idx) = torch.max(out, 1, keepdim=True)
            (acc, num_acc) = recon_metrics_R.accuracy(out, label)
            total_eval_sample += num_acc
            total_eval_correct += acc
    print('total acc: ', total_eval_correct.__float__() / torch.tensor(total_eval_sample).float())