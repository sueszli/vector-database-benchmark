"""
DCGAN - Raw PyTorch Implementation

Code adapted from the official PyTorch DCGAN tutorial:
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""
import os
import random
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils
from torchvision.datasets import CelebA
dataroot = 'data/'
workers = os.cpu_count()
batch_size = 128
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 5
lr = 0.0002
beta1 = 0.5
num_gpus = 1

def main():
    if False:
        print('Hello World!')
    seed = 999
    print('Random Seed: ', seed)
    random.seed(seed)
    torch.manual_seed(seed)
    dataset = CelebA(root=dataroot, split='all', download=True, transform=transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    if torch.cuda.is_available() and num_gpus > 0:
        device = torch.device('cuda:0')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    output_dir = Path('outputs-torch', time.strftime('%Y%m%d-%H%M%S'))
    output_dir.mkdir(parents=True, exist_ok=True)
    real_batch = next(iter(dataloader))
    torchvision.utils.save_image(real_batch[0][:64], output_dir / 'sample-data.png', padding=2, normalize=True)
    generator = Generator().to(device)
    if device.type == 'cuda' and num_gpus > 1:
        generator = nn.DataParallel(generator, list(range(num_gpus)))
    generator.apply(weights_init)
    discriminator = Discriminator().to(device)
    if device.type == 'cuda' and num_gpus > 1:
        discriminator = nn.DataParallel(discriminator, list(range(num_gpus)))
    discriminator.apply(weights_init)
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    real_label = 1.0
    fake_label = 0.0
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    losses_g = []
    losses_d = []
    iteration = 0
    for epoch in range(num_epochs):
        for (i, data) in enumerate(dataloader, 0):
            discriminator.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = discriminator(real_cpu).view(-1)
            err_d_real = criterion(output, label)
            err_d_real.backward()
            d_x = output.mean().item()
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake.detach()).view(-1)
            err_d_fake = criterion(output, label)
            err_d_fake.backward()
            d_g_z1 = output.mean().item()
            err_d = err_d_real + err_d_fake
            optimizer_d.step()
            generator.zero_grad()
            label.fill_(real_label)
            output = discriminator(fake).view(-1)
            err_g = criterion(output, label)
            err_g.backward()
            d_g_z2 = output.mean().item()
            optimizer_g.step()
            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}]\tLoss_D: {err_d.item():.4f}\tLoss_G: {err_g.item():.4f}\tD(x): {d_x:.4f}\tD(G(z)): {d_g_z1:.4f} / {d_g_z2:.4f}')
            losses_g.append(err_g.item())
            losses_d.append(err_d.item())
            if iteration % 500 == 0 or (epoch == num_epochs - 1 and i == len(dataloader) - 1):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                torchvision.utils.save_image(fake, output_dir / f'fake-{iteration:04d}.png', padding=2, normalize=True)
            iteration += 1

def weights_init(m):
    if False:
        while True:
            i = 10
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.main = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True), nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True), nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True), nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True), nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, input):
        if False:
            print('Hello World!')
        return self.main(input)

class Discriminator(nn.Module):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.main = nn.Sequential(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False), nn.Sigmoid())

    def forward(self, input):
        if False:
            return 10
        return self.main(input)
if __name__ == '__main__':
    main()