import os
import numpy as np
import torch
import torch.nn.functional as F
from .gpen_model import FullGenerator

class GPEN(object):

    def __init__(self, model_path, size=512, channel_multiplier=2, device=torch.device('cpu')):
        if False:
            return 10
        self.mfile = model_path
        self.n_mlp = 8
        self.resolution = size
        self.device = device
        self.load_model(channel_multiplier)

    def load_model(self, channel_multiplier=2):
        if False:
            print('Hello World!')
        self.model = FullGenerator(self.resolution, 512, self.n_mlp, channel_multiplier).to(self.device)
        pretrained_dict = torch.load(self.mfile)
        self.model.load_state_dict(pretrained_dict)
        self.model.eval()

    def process(self, im):
        if False:
            return 10
        preds = []
        imt = self.img2tensor(im)
        imt = F.interpolate(imt, (self.resolution, self.resolution))
        with torch.no_grad():
            (img_out, __) = self.model(imt)
        face = self.tensor2img(img_out)
        return (face, preds)

    def img2tensor(self, img):
        if False:
            for i in range(10):
                print('nop')
        img_t = torch.from_numpy(img).to(self.device)
        img_t = (img_t / 255.0 - 0.5) / 0.5
        img_t = img_t.permute(2, 0, 1).unsqueeze(0).flip(1)
        return img_t

    def tensor2img(self, image_tensor, pmax=255.0, imtype=np.uint8):
        if False:
            return 10
        image_tensor = image_tensor * 0.5 + 0.5
        image_tensor = image_tensor.squeeze(0).permute(1, 2, 0).flip(2)
        image_numpy = np.clip(image_tensor.float().cpu().numpy(), 0, 1) * pmax
        return image_numpy.astype(imtype)