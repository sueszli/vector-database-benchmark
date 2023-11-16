import torch
from PIL import Image, ImageFile
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
from os import path
from os import listdir
from six.moves import cPickle
import operator
import math
import Utils
ImageFile.LOAD_TRUNCATED_IMAGES = True

def image_to_tensor(img_path):
    if False:
        return 10
    '\n    As per Pytorch documentations: All pre-trained models expect input images normalized in the same way,\n    i.e. mini-batches of 3-channel RGB images\n    of shape (3 x H x W), where H and W are expected to be at least 224.\n    The images have to be loaded in to a range of [0, 1] and\n    then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].\n    You can use the following transform to normalize:\n    '
    img = Image.open(img_path).convert('RGB')
    transformations = transforms.Compose([transforms.Resize(size=224), transforms.CenterCrop((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image_tensor = transformations(img)[:3, :, :].unsqueeze(0)
    return image_tensor

def load_model(model_path):
    if False:
        return 10
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(2048, 133)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def extract_nn_features(pathIn, model):
    if False:
        return 10
    data = dict()
    if path.exists(pathIn):
        if path.isdir(pathIn):
            print(pathIn)
            dirs = listdir(pathIn)
            for dir in dirs:
                dir_path = path.join(pathIn, dir)
                if path.isdir(dir_path):
                    print(dir_path)
                    imgs = listdir(dir_path)
                    for f in imgs:
                        image_path = path.join(pathIn, dir, f)
                        print(image_path)
                        feature = get_feature(image_path, model)
                        data_feature = dict()
                        data_feature['feature'] = feature
                        data[image_path] = data_feature
    return data

def get_feature(image_path, model):
    if False:
        i = 10
        return i + 15
    image_tf = image_to_tensor(image_path)
    output = model(image_tf)
    pred = output.data.max(1, keepdim=True)[1]
    return pred.numpy()[0][0]

def get_query(image_path, model):
    if False:
        print('Hello World!')
    query = get_feature(image_path, model)
    return query

def calc_distance(features, query):
    if False:
        return 10
    return math.sqrt((features - query) ** 2)

def predict(image_path=None):
    if False:
        return 10
    model = load_model('model_transfer.pt')
    with open('nn_features.pickle', 'rb') as handle:
        data = cPickle.load(handle)
    if image_path == None:
        image_path = Utils.get_random()
    query = get_query(image_path, model)
    distances = dict()
    for id in data.keys():
        dist = calc_distance(data[id]['feature'], query)
        distances[id] = dist
    distances = sorted(distances.items(), key=operator.itemgetter(1))
    result_images = []
    for i in range(6):
        result_images.append(distances[i][0])
    return (image_path, result_images)

def get_train_features():
    if False:
        for i in range(10):
            print('nop')
    model = load_model('model_transfer.pt')
    train_src = 'dogImages/train'
    nn_features = extract_nn_features(train_src, model)
    print('Creating pickle..')
    with open('nn_features.pickle', 'wb') as handle:
        cPickle.dump(nn_features, handle, protocol=cPickle.HIGHEST_PROTOCOL)
if __name__ == '__main__':
    predict()