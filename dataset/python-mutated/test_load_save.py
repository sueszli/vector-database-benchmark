import torch.nn.functional as F
import torch, os, io
import base64
import torch.nn as nn
import unittest
import random
from bigdl.nano.pytorch.patching import patch_encryption
import torch.optim as optim

class TheModelClass(nn.Module):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class linearModel(nn.Module):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(linearModel, self).__init__()
        self.linear = nn.Linear(1, 1)
        self.linear.weight.data.fill_(1.245)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        out = self.linear(x)
        return out

def _create_random(length) -> str:
    if False:
        i = 10
        return i + 15
    ALPHABET = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    chars = []
    for i in range(length):
        chars.append(random.choice(ALPHABET))
    key = ''.join(chars)
    key_bytes = key.encode('ascii')
    base64_str = base64.b64encode(key_bytes)
    print(len(base64_str), flush=True)
    return base64_str
encryption_key = _create_random(32)

class TestModelSaveLoad(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        patch_encryption()

    def test_save_load_to_buf(self):
        if False:
            print('Hello World!')
        model = linearModel()
        encrypted_buf = io.BytesIO()
        expected_buf = io.BytesIO()
        torch.old_save(model.state_dict(), expected_buf)
        expected_buf.seek(0)
        expected_state = torch.load(expected_buf)
        torch.save(model.state_dict(), encrypted_buf, encryption_key=encryption_key)
        our_state_dict = torch.load(encrypted_buf, decryption_key=encryption_key)
        self.assertEqual(our_state_dict, expected_state)

    def test_save_load_to_file(self):
        if False:
            for i in range(10):
                print('nop')
        model = linearModel()
        torch.save(model.state_dict(), 'testsave.pt', encryption_key=encryption_key)
        model.linear.weight.data.fill_(1.11)
        model.load_state_dict(torch.load('testsave.pt', decryption_key=encryption_key))
        self.assertEqual(model.linear.weight.data[0], 1.245)

    def test_save_load_buf2(self):
        if False:
            while True:
                i = 10
        model = linearModel()
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf, encryption_key=encryption_key)
        model.linear.weight.data.fill_(1.11)
        model.load_state_dict(torch.load(buf, decryption_key=encryption_key))
        self.assertEqual(model.linear.weight.data[0], 1.245)

    def test_multi_save(self):
        if False:
            i = 10
            return i + 15
        model = TheModelClass()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        torch.save({'epoch': 5, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': 1.842}, 'checkpoint.pt', encryption_key=encryption_key)
        checkpoint = torch.load('checkpoint.pt', decryption_key=encryption_key)
        self.assertEqual(checkpoint['epoch'], 5)
        self.assertEqual(checkpoint['loss'], 1.842)
        self.assertEqual(optimizer.state_dict(), checkpoint['optimizer_state_dict'])
        for param_tensor in model.state_dict():
            self.assertTrue(torch.equal(model.state_dict()[param_tensor], checkpoint['model_state_dict'][param_tensor]))

    def test_without_keys(self):
        if False:
            while True:
                i = 10
        model = TheModelClass()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        torch.save({'epoch': 5, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': 1.842}, 'checkpoint.pt')
        checkpoint = torch.load('checkpoint.pt')
        self.assertEqual(checkpoint['epoch'], 5)
        self.assertEqual(checkpoint['loss'], 1.842)
        self.assertEqual(optimizer.state_dict(), checkpoint['optimizer_state_dict'])
        for param_tensor in model.state_dict():
            self.assertTrue(torch.equal(model.state_dict()[param_tensor], checkpoint['model_state_dict'][param_tensor]))