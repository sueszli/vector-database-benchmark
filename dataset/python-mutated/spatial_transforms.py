import random
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
from PIL import Image

class Compose(transforms.Compose):

    def randomize_parameters(self):
        if False:
            return 10
        for t in self.transforms:
            t.randomize_parameters()

class ToTensor(transforms.ToTensor):

    def randomize_parameters(self):
        if False:
            print('Hello World!')
        pass

class Normalize(transforms.Normalize):

    def randomize_parameters(self):
        if False:
            while True:
                i = 10
        pass

class ScaleValue(object):

    def __init__(self, s):
        if False:
            print('Hello World!')
        self.s = s

    def __call__(self, tensor):
        if False:
            return 10
        tensor *= self.s
        return tensor

    def randomize_parameters(self):
        if False:
            print('Hello World!')
        pass

class Resize(transforms.Resize):

    def randomize_parameters(self):
        if False:
            while True:
                i = 10
        pass

class Scale(transforms.Scale):

    def randomize_parameters(self):
        if False:
            return 10
        pass

class CenterCrop(transforms.CenterCrop):

    def randomize_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class CornerCrop(object):

    def __init__(self, size, crop_position=None, crop_positions=['c', 'tl', 'tr', 'bl', 'br']):
        if False:
            while True:
                i = 10
        self.size = size
        self.crop_position = crop_position
        self.crop_positions = crop_positions
        if crop_position is None:
            self.randomize = True
        else:
            self.randomize = False
        self.randomize_parameters()

    def __call__(self, img):
        if False:
            i = 10
            return i + 15
        image_width = img.size[0]
        image_height = img.size[1]
        (h, w) = (self.size, self.size)
        if self.crop_position == 'c':
            i = int(round((image_height - h) / 2.0))
            j = int(round((image_width - w) / 2.0))
        elif self.crop_position == 'tl':
            i = 0
            j = 0
        elif self.crop_position == 'tr':
            i = 0
            j = image_width - self.size
        elif self.crop_position == 'bl':
            i = image_height - self.size
            j = 0
        elif self.crop_position == 'br':
            i = image_height - self.size
            j = image_width - self.size
        img = F.crop(img, i, j, h, w)
        return img

    def randomize_parameters(self):
        if False:
            while True:
                i = 10
        if self.randomize:
            self.crop_position = self.crop_positions[random.randint(0, len(self.crop_positions) - 1)]

    def __repr__(self):
        if False:
            while True:
                i = 10
        return self.__class__.__name__ + '(size={0}, crop_position={1}, randomize={2})'.format(self.size, self.crop_position, self.randomize)

class RandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def __init__(self, p=0.5):
        if False:
            while True:
                i = 10
        super().__init__(p)
        self.randomize_parameters()

    def __call__(self, img):
        if False:
            print('Hello World!')
        '\n        Args:\n            img (PIL.Image): Image to be flipped.\n        Returns:\n            PIL.Image: Randomly flipped image.\n        '
        if self.random_p < self.p:
            return F.hflip(img)
        return img

    def randomize_parameters(self):
        if False:
            print('Hello World!')
        self.random_p = random.random()

class MultiScaleCornerCrop(object):

    def __init__(self, size, scales, crop_positions=['c', 'tl', 'tr', 'bl', 'br'], interpolation=Image.BILINEAR):
        if False:
            print('Hello World!')
        self.size = size
        self.scales = scales
        self.interpolation = interpolation
        self.crop_positions = crop_positions
        self.randomize_parameters()

    def __call__(self, img):
        if False:
            print('Hello World!')
        short_side = min(img.size[0], img.size[1])
        crop_size = int(short_side * self.scale)
        self.corner_crop.size = crop_size
        img = self.corner_crop(img)
        return img.resize((self.size, self.size), self.interpolation)

    def randomize_parameters(self):
        if False:
            while True:
                i = 10
        self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        crop_position = self.crop_positions[random.randint(0, len(self.crop_positions) - 1)]
        self.corner_crop = CornerCrop(None, crop_position)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return self.__class__.__name__ + '(size={0}, scales={1}, interpolation={2})'.format(self.size, self.scales, self.interpolation)

class RandomResizedCrop(transforms.RandomResizedCrop):

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation=Image.BILINEAR):
        if False:
            while True:
                i = 10
        super().__init__(size, scale, ratio, interpolation)
        self.randomize_parameters()

    def __call__(self, img):
        if False:
            while True:
                i = 10
        if self.randomize:
            self.random_crop = self.get_params(img, self.scale, self.ratio)
            self.randomize = False
        (i, j, h, w) = self.random_crop
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def randomize_parameters(self):
        if False:
            print('Hello World!')
        self.randomize = True

class ColorJitter(transforms.ColorJitter):

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        if False:
            return 10
        super().__init__(brightness, contrast, saturation, hue)
        self.randomize_parameters()

    def __call__(self, img):
        if False:
            i = 10
            return i + 15
        if self.randomize:
            self.transform = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
            self.randomize = False
        return self.transform(img)

    def randomize_parameters(self):
        if False:
            print('Hello World!')
        self.randomize = True

class PickFirstChannels(object):

    def __init__(self, n):
        if False:
            print('Hello World!')
        self.n = n

    def __call__(self, tensor):
        if False:
            print('Hello World!')
        return tensor[:self.n, :, :]

    def randomize_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        pass