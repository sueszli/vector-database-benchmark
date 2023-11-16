import torch

class TestNnModule(torch.nn.Module):

    def __init__(self, nz=6, ngf=9, nc=3):
        if False:
            while True:
                i = 10
        super().__init__()
        self.main = torch.nn.Sequential(torch.nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), torch.nn.BatchNorm2d(ngf * 8), torch.nn.ReLU(True), torch.nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), torch.nn.BatchNorm2d(ngf * 4), torch.nn.ReLU(True), torch.nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), torch.nn.BatchNorm2d(ngf * 2), torch.nn.ReLU(True), torch.nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), torch.nn.BatchNorm2d(ngf), torch.nn.ReLU(True), torch.nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), torch.nn.Tanh())

    def forward(self, input):
        if False:
            while True:
                i = 10
        return self.main(input)