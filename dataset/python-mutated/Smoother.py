import math
import numpy as np
import torch
import torch.nn as nn
from modelscope.models.cv.video_stabilization.utils.IterativeSmooth import generateSmooth

class Smoother(nn.Module):

    def __init__(self, inplanes=2, embeddingSize=64, hiddenSize=64, kernel=5):
        if False:
            while True:
                i = 10
        super(Smoother, self).__init__()
        self.embedding = nn.Sequential(nn.Linear(inplanes, embeddingSize), nn.ReLU())
        self.pad = kernel // 2
        self.conv1 = nn.Conv3d(embeddingSize, embeddingSize, (kernel, 3, 3), padding=(self.pad, 1, 1))
        self.conv3 = nn.Conv3d(embeddingSize, embeddingSize, (kernel, 3, 3), padding=(self.pad, 1, 1))
        self.conv2 = nn.Conv3d(embeddingSize, embeddingSize, (kernel, 3, 3), padding=(self.pad, 1, 1))
        self.decoder = nn.Linear(embeddingSize, 12, bias=True)
        self.scale = nn.Linear(embeddingSize, 1, bias=True)
        self.activation = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.generateSmooth = generateSmooth

    def forward(self, trajectory):
        if False:
            i = 10
            return i + 15
        '\n        @param trajectory: Unstable trajectory with shape [B, 2, T, H, W]\n\n        @return kernel: dynamic smooth kernel with shape [B, 12, T, H, W]\n        '
        trajectory = trajectory.permute(0, 2, 3, 4, 1)
        embedding_trajectory = self.embedding(trajectory).permute(0, 4, 1, 2, 3)
        hidden = embedding_trajectory
        hidden = self.relu(self.conv1(hidden))
        hidden = self.relu(self.conv3(hidden))
        hidden = self.relu(self.conv2(hidden))
        kernel = self.activation(self.decoder(hidden.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3))
        kernel = self.scale(hidden.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3) * kernel
        return kernel

    def inference(self, x_paths, y_paths, repeat=50):
        if False:
            return 10
        '\n        @param x_paths: Unstable trajectory in x direction, [B, T, H, W]\n        @param y_paths: Unstable trajectory in y direction, [B, T, H, W]\n        @param repeat: iterations for smoother, int\n\n        @return smooth_x: Smoothed trajectory in x direction, [B, T, H, W]\n        @return smooth_y: Smoothed trajectory in y direction, [B, T, H, W]\n        '
        path = np.concatenate([np.expand_dims(x_paths, -1), np.expand_dims(y_paths, -1)], -1)
        min_v = np.min(path, keepdims=True)
        path = path - min_v
        max_v = np.max(path, keepdims=True) + 1e-05
        path = path / max_v
        path = np.transpose(np.expand_dims(path, 0), (0, 4, 3, 1, 2))
        path_t = torch.from_numpy(path.astype(np.float32)).cuda()
        kernel_t = self.forward(path_t)
        (smooth_x, smooth_y) = self.KernelSmooth(kernel_t, path_t, repeat)
        smooth_x = smooth_x.cpu().squeeze().permute(1, 2, 0).numpy() * max_v + min_v
        smooth_y = smooth_y.cpu().squeeze().permute(1, 2, 0).numpy() * max_v + min_v
        return (smooth_x, smooth_y)

    def KernelSmooth(self, kernel, path, repeat=20):
        if False:
            for i in range(10):
                print('nop')
        if kernel is None:
            smooth_x = self.generateSmooth(path[:, 0:1, :, :, :], None, repeat)
            smooth_y = self.generateSmooth(path[:, 1:2, :, :, :], None, repeat)
        else:
            smooth_x = self.generateSmooth(path[:, 0:1, :, :, :], kernel[:, 0:6, :, :, :], repeat)
            smooth_y = self.generateSmooth(path[:, 1:2, :, :, :], kernel[:, 6:12, :, :, :], repeat)
        return (smooth_x, smooth_y)