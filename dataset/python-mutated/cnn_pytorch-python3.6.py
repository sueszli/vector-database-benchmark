"""
Created on 2017-12-18
Update  on 2018-03-27
Author: 片刻
Github: https://github.com/apachecn/kaggle
Result: 
    BATCH_SIZE = 10 and EPOCH = 10; [10,  4000] loss: 0.069
    BATCH_SIZE = 10 and EPOCH = 15; [10,  4000] loss: 0.069
"""
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os.path
data_dir = '/opt/data/kaggle/getting-started/digit-recognizer/'

class CustomedDataSet(Dataset):

    def __init__(self, train=True):
        if False:
            for i in range(10):
                print('nop')
        self.train = train
        if self.train:
            trainX = pd.read_csv(os.path.join(data_dir, 'input/train.csv'))
            trainY = trainX.label.as_matrix().tolist()
            trainX = trainX.drop('label', axis=1).as_matrix().reshape(trainX.shape[0], 1, 28, 28)
            self.datalist = trainX
            self.labellist = trainY
        else:
            testX = pd.read_csv(os.path.join(data_dir, 'input/test.csv'))
            self.testID = testX.index
            testX = testX.as_matrix().reshape(testX.shape[0], 1, 28, 28)
            self.datalist = testX

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        if self.train:
            return (torch.Tensor(self.datalist[index].astype(float)), self.labellist[index])
        else:
            return torch.Tensor(self.datalist[index].astype(float))

    def __len__(self):
        if False:
            print('Hello World!')
        return self.datalist.shape[0]
train_data = CustomedDataSet()
test_data = CustomedDataSet(train=False)
BATCH_SIZE = 150
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

class CNN(nn.Module):

    def __init__(self):
        if False:
            return 10
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return (output, x)
cnn = CNN()
LR = 0.001
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
print(u'开始训练')
EPOCH = 5
for epoch in range(EPOCH):
    running_loss = 0.0
    for (step, (x, y)) in enumerate(train_loader):
        b_x = Variable(x)
        b_y = Variable(y)
        output = cnn(b_x)[0]
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        if step % 500 == 499:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, step + 1, running_loss / 500))
            running_loss = 0.0
print('Finished Training')
ans = torch.LongTensor()
for img in test_loader:
    img = Variable(img)
    outputs = cnn(img)
    (_, predicted) = torch.max(outputs[0], 1)
    ans = torch.cat([ans, predicted.data], 0)
testLabel = ans.numpy()
submission_df = pd.DataFrame(data={'ImageId': test_data.testID + 1, 'Label': testLabel})
submission_df.to_csv(os.path.join(data_dir, 'output/Result_pytorch_CNN.csv'), columns=['ImageId', 'Label'], index=False)