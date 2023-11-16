import argparse
import logging
import sys
import nni
from nni.algorithms.hpo.networkmorphism_tuner.graph import json_to_graph
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import utils
log_format = '%(asctime)s %(message)s'
logging.basicConfig(filename='networkmorphism.log', filemode='a', level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger('cifar10-network-morphism-pytorch')

def get_args():
    if False:
        print('Hello World!')
    ' get args from command line\n    '
    parser = argparse.ArgumentParser('cifar10')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer')
    parser.add_argument('--epochs', type=int, default=200, help='epoch limit')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=8, help='cutout length')
    parser.add_argument('--model_path', type=str, default='./', help='Path to save the destination model')
    return parser.parse_args()
trainloader = None
testloader = None
net = None
criterion = None
optimizer = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0.0
args = get_args()

def build_graph_from_json(ir_model_json):
    if False:
        print('Hello World!')
    'build model from json representation\n    '
    graph = json_to_graph(ir_model_json)
    logging.debug(graph.operation_history)
    model = graph.produce_torch_model()
    return model

def parse_rev_args(receive_msg):
    if False:
        for i in range(10):
            print('nop')
    ' parse reveive msgs to global variable\n    '
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer
    logger.debug('Preparing data..')
    (transform_train, transform_test) = utils.data_transforms_cifar10(args)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    logger.debug('Building model..')
    net = build_graph_from_json(receive_msg)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    if device == 'cuda' and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    if args.optimizer == 'Adadelta':
        optimizer = optim.Adadelta(net.parameters(), lr=args.learning_rate)
    if args.optimizer == 'Adagrad':
        optimizer = optim.Adagrad(net.parameters(), lr=args.learning_rate)
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    if args.optimizer == 'Adamax':
        optimizer = optim.Adamax(net.parameters(), lr=args.learning_rate)
    if args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(net.parameters(), lr=args.learning_rate)
    return 0

def train(epoch):
    if False:
        for i in range(10):
            print('nop')
    ' train model on each epoch in trainset\n    '
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer
    logger.debug('Epoch: %d', epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for (batch_idx, (inputs, targets)) in enumerate(trainloader):
        (inputs, targets) = (inputs.to(device), targets.to(device))
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        (_, predicted) = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100.0 * correct / total
        logger.debug('Loss: %.3f | Acc: %.3f%% (%d/%d)', train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total)
    return acc

def test(epoch):
    if False:
        i = 10
        return i + 15
    ' eval model on each epoch in testset\n    '
    global best_acc
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer
    logger.debug('Eval on epoch: %d', epoch)
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for (batch_idx, (inputs, targets)) in enumerate(testloader):
            (inputs, targets) = (inputs.to(device), targets.to(device))
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            (_, predicted) = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100.0 * correct / total
            logger.debug('Loss: %.3f | Acc: %.3f%% (%d/%d)', test_loss / (batch_idx + 1), 100.0 * correct / total, correct, total)
    acc = 100.0 * correct / total
    if acc > best_acc:
        best_acc = acc
    return (acc, best_acc)
if __name__ == '__main__':
    try:
        RCV_CONFIG = nni.get_next_parameter()
        logger.debug(RCV_CONFIG)
        parse_rev_args(RCV_CONFIG)
        train_acc = 0.0
        best_acc = 0.0
        early_stop = utils.EarlyStopping(mode='max')
        for ep in range(args.epochs):
            train_acc = train(ep)
            (test_acc, best_acc) = test(ep)
            nni.report_intermediate_result(test_acc)
            logger.debug(test_acc)
            if early_stop.step(test_acc):
                break
        nni.report_final_result(best_acc)
    except Exception as exception:
        logger.exception(exception)
        raise