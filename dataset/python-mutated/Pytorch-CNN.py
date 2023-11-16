"""
 PyTorch 版本的 CNN 实现 Dogs vs Cats
"""
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms as T
from models import AlexNet
'\n参考链接：\nhttps://github.com/pytorch/examples/blob/27e2a46c1d1505324032b1d94fc6ce24d5b67e97/imagenet/main.py\n'
'\n这里我们使用了 torch.utils.data 中的一些函数，比如 Dataset\nclass torch.utils.data.Dataset\n表示 Dataset 的抽象类\n所有其他的数据集都应继承该类。所有子类都应该重写 __len__ ，提供数据集大小的方法，和 __getitem__ ，支持从 0 到 len(self) 整数索引的方法\n'

class GetData(data.Dataset):

    def __init__(self, root, transforms=None, train=True, test=Flase):
        if False:
            print('Hello World!')
        '\n        Desc:\n            获取全部的数据，并根据我们的要求，将数据划分为 训练、验证、测试数据集\n        Args:\n            self --- none\n            root --- 数据存在路径\n            transforms --- 对数据的转化，这里默认是 None\n            train ---- 标注是否是训练集\n            test  ---- 标注是否是测试集\n        Returns:\n            None\n        '
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))
        imgs_num = len(imgs)
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7 * imgs_num)]
        else:
            self.imgs = imgs[int(0.7 * imgs_num):]
        if transforms is None:
            '\n            几个常见的 transforms 用的转换：\n            1、数据归一化 --- Normalize(mean, std) 是通过下面公式实现数据归一化 channel = (channel-mean)/std\n            2、class torchvision.transforms.Resize(size, interpolation=2) 将输入的 PIL 图像调整为给定的大小\n            3、class torchvision.transforms.CenterCrop(size) 在中心裁剪给定的 PIL 图像，参数 size 是期望的输出大小\n            4、ToTensor() 是将 PIL.Image(RGB) 或者 numpy.ndarray(H X W X C) 从 0 到 255 的值映射到 0~1 的范围内，并转化为 Tensor 形式\n            5、transforms.Compose() 这个是将多个 transforms 组合起来使用\n            '
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if self.test or not train:
                self.trainsforms = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), normalize])
            else:
                self.transforms = T.Compose([T.Resize(256), T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(), normalize])

    def __len__(self):
        if False:
            i = 10
            return i + 15
        '\n        Desc:\n            继承 Dataset 基类，重写 __len__ 方法，提供数据集的大小\n        Args:\n            self --- 无\n        Return:\n            数据集的长度\n        '
        return len(self.imgs)

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        '\n        Desc:\n            继承 Dataset 基类，重写 __getitem__ 方法，支持整数索引，范围从 0 到 len(self) \n            返回一张图片的数据\n            对于测试集，没有label，返回图片 id，如 985.jpg 返回 985\n            对于训练集，是具有 label ，返回图片 id ，以及相对应的 label，如 dog.211.jpg 返回 id 为 211，label 为 dog\n        Args:\n            self --- none\n            index --- 索引\n        Return:\n            返回一张图片的数据\n            对于测试集，没有label，返回图片 id，如 985.jpg 返回 985\n            对于训练集，是具有 label ，返回图片 id ，以及相对应的 label，如 dog.211.jpg 返回 id 为 211，label 为 dog\n        '
        img_path = self.imgs[index]
        if self.test:
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return (data, label)
train_path = 'D:/dataset/dogs-vs-cats/train'
train_dataset = GetData(train_path, train=True)
load_train = data.DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=1)
'\nutils.data.DataLoader() 解析\nclass torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)\n数据加载器。组合数据集和采样器，并在数据集上提供单个或多个进程迭代器。\n参数：\ndataset(Dataset) --- 从这之中加载数据的数据集。\nbatch_size (int, 可选) --- 每个 batch 加载多少个样本（默认值为：1）\nshuffle (bool, 可选) --- 设置为 True 时，会在每个 epoch 时期重新组合数据（默认值：False）\nsampler (Sampler, 可选) --- 定义从数据集中抽取样本的策略。如果指定，那么 shuffle 必须是 False 。\nbatch_sampler (Sampler, 可选) --- 类似采样器，但一次返回一批量的索引（index）。与 batch_size, shuffle, sampler 和 drop_last 相互排斥。\nnum_workers (int, 可选) --- 设置有多少个子进程用于数据加载。0 表示数据将在主进程中加载。（默认：0）\ncollate_fn (callable, 可选) --- 合并样本列表以形成 mini-batch \npin_memory (bool, 可选) ---  如果为 True，数据加载器会在 tensors 返回之前将 tensors 复制到 CUDA 固定内存中。\ndrop_last (bool, 可选) --- 如果 dataset size （数据集大小）不能被 batch size （批量大小）整除，则设置为 True 以删除最后一个 incomplete batch（未完成批次）。\n                          如果设置为 False 和 dataset size（数据集大小）不能被 batch size（批量大小）整除，则最后一批将会更小。（默认：False）\ntimeout (numeric, 可选) --- 如果是正值，则为从 worker 收集 batch 的超时值。应该始终是非负的。（默认：0）\nworker_init_fn (callable, 可选) --- 如果不是 None，那么将在每个工人子进程上使用 worker id（在 [0，num_workers - 1] 中的 int）作为输入，在 seeding 和加载数据之前调用这个子进程。（默认：无）\n'
test_path = 'D:/dataset/dogs-vs-cats/test1'
test_path = GetData(test_path, test=True)
loader_test = data.DataLoader(test_dataset, batch_size=3, shuffle=True, num_workers=1)
cnn = AlexNet()
print(cnn)
'\n1、torch.optim 是一个实现各种优化算法的软件包。\n比如我们这里使用的就是 Adam() 这个方法\nclass torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False) 这个类就实现了 adam 算法。\nparams(iterable) --- 可迭代的参数来优化或取消定义参数组\nlr(float, 可选) --- 学习率（默认值 1e-3）\nbeta(Tuple[float, float], 可选) --- 用于计算梯度及其平方的运行平均值的系数（默认值：（0.9，0.999））\neps (float, 可选) ---- 添加到分母以提高数值稳定性（默认值：1e-8）\nweight_decay (float, 可选) --- 权重衰减（L2 惩罚）（默认值：0）\namsgrad (boolean, 可选) ---- 是否使用该算法的AMSGrad变体来自论文关于 Adam 和 Beyond 的融合  \n\n\n2、还有这里我们使用的损失函数 \nclass torch.nn.CrossEntropyLoss(weight=None, size_average=True, ignore_index=-100, reduce=True)\n交叉熵损失函数\n具体的请看：http://pytorch.apachecn.org/cn/docs/0.3.0/nn.html   \n'
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.005, betas=(0.9, 0.99))
loss_func = nn.CrossEntropyLoss()
EPOCH = 10
for epoch in range(EPOCH):
    num = 0
    for (step, (x, y)) in enumerate(loader_train):
        b_x = Variable(x)
        b_y = Variable(y)
        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            num += 1
            for (_, (x_t, y_test)) in enumerate(loader_test):
                x_test = Variable(x_t)
                test_output = cnn(x_test)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                accuracy = sum(pred_y == y_test) / float(y_test.size(0))
                print('Epoch: ', epoch, '| Num: ', num, '| Step: ', step, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.4f' % accuracy)