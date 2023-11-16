import torch as t
import time

class BasicModule(t.nn.Module):
    """
    封装了nn.Module,主要是提供了save和load两个方法
    """

    def __init__(self):
        if False:
            return 10
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        if False:
            return 10
        '\n        可加载指定路径的模型\n        '
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        if False:
            print('Hello World!')
        '\n        保存模型，默认使用“模型名字+时间”作为文件名\n        '
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name

class Flat(t.nn.Module):
    """
    把输入reshape成（batch_size,dim_length）
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(Flat, self).__init__()

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        return x.view(x.size(0), -1)