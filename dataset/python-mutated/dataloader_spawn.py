from fastai.vision.all import *

def get_data(url, presize, resize):
    if False:
        for i in range(10):
            print('nop')
    path = untar_data(url)
    return DataBlock(blocks=(ImageBlock, CategoryBlock), get_items=get_image_files, splitter=GrandparentSplitter(valid_name='val'), get_y=parent_label, item_tfms=Resize(presize), batch_tfms=aug_transforms(min_scale=0.5, size=resize)).dataloaders(path, bs=128)

def block(ni, nf):
    if False:
        return 10
    return ConvLayer(ni, nf, stride=2)

def get_model():
    if False:
        for i in range(10):
            print('nop')
    return nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 128), block(128, 256), nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(256, dls.c))

def get_learner(dls, m):
    if False:
        while True:
            i = 10
    return Learner(dls, m, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    dls = get_data(URLs.IMAGENETTE_160, 160, 128)
    resnet_model = get_model()
    learn = get_learner(dls, resnet_model)
    learn.lr_find()