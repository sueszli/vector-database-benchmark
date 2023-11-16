import torch
import torch.nn as nn
from .decoder import Decoder
from .encoder import Encoder

class BtsModel(nn.Module):
    """Depth estimation model bts, implemented from paper https://arxiv.org/pdf/1907.10326.pdf.
        The network utilizes novel local planar guidance layers located at multiple stage in the decoding phase.
        The bts model is composed with encoder and decoder, an encoder for dense feature extraction and a decoder
        for predicting the desired depth.
    """

    def __init__(self, focal=715.0873):
        if False:
            print('Hello World!')
        'initial bts model\n\n        Args:\n            focal (float): focal length, pictures that do not work are input according to\n                the camera setting value at the time of shooting\n        '
        super(BtsModel, self).__init__()
        self.focal = focal
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x, focal=None):
        if False:
            while True:
                i = 10
        'forward to estimation depth\n\n        Args:\n            x (Tensor): input image data\n            focal (float): The focal length when the picture is taken. By default, the focal length\n                of the data set when the model is created is used\n\n        Returns:\n            Tensor: Depth estimation image\n        '
        focal_run = focal if focal else self.focal
        skip_feat = self.encoder(x)
        depth = self.decoder(skip_feat, torch.tensor(focal_run).cuda())
        return depth