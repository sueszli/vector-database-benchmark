import os
import tempfile
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed.fleet import auto
from paddle.io import Dataset
paddle.enable_static()
batch_size = 2
batch_num = 10
hidden_size = 1024
sequence_len = 512
image_size = hidden_size
class_num = 10
paddle.seed(44)

class MyDataset(Dataset):

    def __init__(self, num_samples):
        if False:
            return 10
        super().__init__()
        self.num_samples = num_samples

    def __getitem__(self, index):
        if False:
            print('Hello World!')
        input = np.random.uniform(size=image_size).astype('float32')
        label = np.random.randint(0, class_num - 1, dtype='int64')
        return (input, label)

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return self.num_samples

class MLPLayer(nn.Layer):

    def __init__(self, hidden_size=1024, intermediate_size=4 * 1024, dropout_ratio=0.1, initializer_range=0.02):
        if False:
            return 10
        super().__init__()
        d_model = hidden_size
        dim_feedforward = intermediate_size
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=initializer_range))
        bias_attr = None
        self.linear0 = nn.Linear(d_model, dim_feedforward, weight_attr, bias_attr=bias_attr)
        self.linear1 = nn.Linear(dim_feedforward, d_model, weight_attr, bias_attr=bias_attr)
        self.linear2 = nn.Linear(d_model, 1, weight_attr, bias_attr=bias_attr)
        self.norm = nn.LayerNorm(d_model, epsilon=1e-05)
        self.dropout = nn.Dropout(dropout_ratio, mode='upscale_in_train')

    def forward(self, input):
        if False:
            while True:
                i = 10
        out = self.norm(input)
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        out = self.linear1(out)
        out = self.dropout(out)
        out = self.linear2(out)
        auto.fetch(out, 'out')
        self.out = out
        return out

def train(fetch):
    if False:
        print('Hello World!')
    mlp = MLPLayer(hidden_size=hidden_size, intermediate_size=4 * hidden_size, dropout_ratio=0.1, initializer_range=0.02)
    loss = paddle.nn.CrossEntropyLoss()
    optimizer = paddle.optimizer.Adam(learning_rate=1e-05, beta1=0.9, beta2=0.999, epsilon=1e-08, grad_clip=None)
    dist_strategy = auto.Strategy()
    dist_strategy.auto_mode = 'semi'
    engine = auto.Engine(mlp, loss, optimizer, paddle.metric.Accuracy(), strategy=dist_strategy)
    train_dataset = MyDataset(batch_num * batch_size)
    engine.fit(train_dataset, batch_size=batch_size)
    eval_dataset = MyDataset(batch_size)
    engine.evaluate(eval_dataset, batch_size=batch_size)
    test_dataset = MyDataset(batch_size)
    engine.predict(test_dataset, batch_size=batch_size)
    temp_dir = tempfile.TemporaryDirectory()
    model_filename = os.path.join(temp_dir.name, 'mlp_inf')
    engine.save(model_filename, training=False)
    temp_dir.cleanup()
if __name__ == '__main__':
    train(True)