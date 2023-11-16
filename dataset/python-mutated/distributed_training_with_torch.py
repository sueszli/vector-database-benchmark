"""
Title: Multi-GPU distributed training with PyTorch
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2023/06/29
Last modified: 2023/06/29
Description: Guide to multi-GPU training for Keras models with PyTorch.
Accelerator: GPU
"""
"\n## Introduction\n\nThere are generally two ways to distribute computation across multiple devices:\n\n**Data parallelism**, where a single model gets replicated on multiple devices or\nmultiple machines. Each of them processes different batches of data, then they merge\ntheir results. There exist many variants of this setup, that differ in how the different\nmodel replicas merge results, in whether they stay in sync at every batch or whether they\nare more loosely coupled, etc.\n\n**Model parallelism**, where different parts of a single model run on different devices,\nprocessing a single batch of data together. This works best with models that have a\nnaturally-parallel architecture, such as models that feature multiple branches.\n\nThis guide focuses on data parallelism, in particular **synchronous data parallelism**,\nwhere the different replicas of the model stay in sync after each batch they process.\nSynchronicity keeps the model convergence behavior identical to what you would see for\nsingle-device training.\n\nSpecifically, this guide teaches you how to use PyTorch's `DistributedDataParallel`\nmodule wrapper to train Keras, with minimal changes to your code,\non multiple GPUs (typically 2 to 16) installed on a single machine (single host,\nmulti-device training). This is the most common setup for researchers and small-scale\nindustry workflows.\n"
"\n## Setup\n\nLet's start by defining the function that creates the model that we will train,\nand the function that creates the dataset we will train on (MNIST in this case).\n"
import os
os.environ['KERAS_BACKEND'] = 'torch'
import torch
import numpy as np
import keras

def get_model():
    if False:
        i = 10
        return i + 15
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = keras.layers.Conv2D(filters=12, kernel_size=3, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization(scale=False, center=True)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(filters=24, kernel_size=6, use_bias=False, strides=2)(x)
    x = keras.layers.BatchNormalization(scale=False, center=True)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=6, padding='same', strides=2, name='large_k')(x)
    x = keras.layers.BatchNormalization(scale=False, center=True)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(10)(x)
    model = keras.Model(inputs, outputs)
    return model

def get_dataset():
    if False:
        for i in range(10):
            print('nop')
    ((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print('x_train shape:', x_train.shape)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    return dataset
"\nNext, let's define a simple PyTorch training loop that targets\na GPU (note the calls to `.cuda()`).\n"

def train_model(model, dataloader, num_epochs, optimizer, loss_fn):
    if False:
        i = 10
        return i + 15
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_loss_count = 0
        for (batch_idx, (inputs, targets)) in enumerate(dataloader):
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_loss_count += 1
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / running_loss_count}')
"\n## Single-host, multi-device synchronous training\n\nIn this setup, you have one machine with several GPUs on it (typically 2 to 16). Each\ndevice will run a copy of your model (called a **replica**). For simplicity, in what\nfollows, we'll assume we're dealing with 8 GPUs, at no loss of generality.\n\n**How it works**\n\nAt each step of training:\n\n- The current batch of data (called **global batch**) is split into 8 different\nsub-batches (called **local batches**). For instance, if the global batch has 512\nsamples, each of the 8 local batches will have 64 samples.\n- Each of the 8 replicas independently processes a local batch: they run a forward pass,\nthen a backward pass, outputting the gradient of the weights with respect to the loss of\nthe model on the local batch.\n- The weight updates originating from local gradients are efficiently merged across the 8\nreplicas. Because this is done at the end of every step, the replicas always stay in\nsync.\n\nIn practice, the process of synchronously updating the weights of the model replicas is\nhandled at the level of each individual weight variable. This is done through a **mirrored\nvariable** object.\n\n**How to use it**\n\nTo do single-host, multi-device synchronous training with a Keras model, you would use\nthe `torch.nn.parallel.DistributedDataParallel` module wrapper.\nHere's how it works:\n\n- We use `torch.multiprocessing.start_processes` to start multiple Python processes, one\nper device. Each process will run the `per_device_launch_fn` function.\n- The `per_device_launch_fn` function does the following:\n    - It uses `torch.distributed.init_process_group` and `torch.cuda.set_device`\n    to configure the device to be used for that process.\n    - It uses `torch.utils.data.distributed.DistributedSampler`\n    and `torch.utils.data.DataLoader` to turn our data into a distributed data loader.\n    - It also uses `torch.nn.parallel.DistributedDataParallel` to turn our model into\n    a distributed PyTorch module.\n    - It then calls the `train_model` function.\n- The `train_model` function will then run in each process, with the model using\na separate device in each process.\n\nHere's the flow, where each step is split into its own utility function:\n"
num_gpu = torch.cuda.device_count()
num_epochs = 2
batch_size = 64
print(f'Running on {num_gpu} GPUs')

def setup_device(current_gpu_index, num_gpus):
    if False:
        print('Hello World!')
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '56492'
    device = torch.device('cuda:{}'.format(current_gpu_index))
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=num_gpus, rank=current_gpu_index)
    torch.cuda.set_device(device)

def cleanup():
    if False:
        i = 10
        return i + 15
    torch.distributed.destroy_process_group()

def prepare_dataloader(dataset, current_gpu_index, num_gpus, batch_size):
    if False:
        print('Hello World!')
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_gpus, rank=current_gpu_index, shuffle=False)
    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size, shuffle=False)
    return dataloader

def per_device_launch_fn(current_gpu_index, num_gpu):
    if False:
        print('Hello World!')
    setup_device(current_gpu_index, num_gpu)
    dataset = get_dataset()
    model = get_model()
    dataloader = prepare_dataloader(dataset, current_gpu_index, num_gpu, batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    model = model.to(current_gpu_index)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[current_gpu_index], output_device=current_gpu_index)
    train_model(ddp_model, dataloader, num_epochs, optimizer, loss_fn)
    cleanup()
'\nTime to start multiple processes:\n'
if __name__ == '__main__':
    torch.multiprocessing.start_processes(per_device_launch_fn, args=(num_gpu,), nprocs=num_gpu, join=True, start_method='fork')
"\nThat's it!\n"