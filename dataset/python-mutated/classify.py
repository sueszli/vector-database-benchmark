from PIL import Image
import argparse
import numpy as np
import json
import torch
from torch.cuda.amp import autocast
import torch.backends.cudnn as cudnn
from image_classification import models
import torchvision.transforms as transforms
from image_classification.models import efficientnet_b0

def available_models():
    if False:
        i = 10
        return i + 15
    models = {m.name: m for m in [efficientnet_b0]}
    return models

def add_parser_arguments(parser):
    if False:
        while True:
            i = 10
    model_names = available_models().keys()
    parser.add_argument('--image-size', default='224', type=int)
    parser.add_argument('--arch', '-a', metavar='ARCH', default='efficientnet-b0', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: efficientnet-b0)')
    parser.add_argument('--precision', metavar='PREC', default='AMP', choices=['AMP', 'FP32'])
    parser.add_argument('--cpu', action='store_true', help='perform inference on CPU')
    parser.add_argument('--image', metavar='<path>', help='path to classified image')

def load_jpeg_from_file(path, image_size, cuda=True):
    if False:
        i = 10
        return i + 15
    img_transforms = transforms.Compose([transforms.Resize(image_size + 32), transforms.CenterCrop(image_size), transforms.ToTensor()])
    img = img_transforms(Image.open(path))
    with torch.no_grad():
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        if cuda:
            mean = mean.cuda()
            std = std.cuda()
            img = img.cuda()
        img = img.float()
        input = img.unsqueeze(0).sub_(mean).div_(std)
    return input

def check_quant_weight_correctness(checkpoint_path, model):
    if False:
        while True:
            i = 10
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = {k[len('module.'):] if k.startswith('module.') else k: v for (k, v) in state_dict.items()}
    quantizers_sd_keys = {f'{n[0]}._amax' for n in model.named_modules() if 'quantizer' in n[0]}
    sd_all_keys = quantizers_sd_keys | set(model.state_dict().keys())
    assert set(state_dict.keys()) == sd_all_keys, f'Passed quantized architecture, but following keys are missing in checkpoint: {list(sd_all_keys - set(state_dict.keys()))}'

def main(args, model_args):
    if False:
        return 10
    imgnet_classes = np.array(json.load(open('./LOC_synset_mapping.json', 'r')))
    try:
        model = available_models()[args.arch](**model_args.__dict__)
    except RuntimeError as e:
        print_in_box('Error when creating model, did you forget to run checkpoint2model script?')
        raise e
    if args.arch in ['efficientnet-quant-b0', 'efficientnet-quant-b4']:
        check_quant_weight_correctness(model_args.pretrained_from_file, model)
    if not args.cpu:
        model = model.cuda()
    model.eval()
    input = load_jpeg_from_file(args.image, args.image_size, cuda=not args.cpu)
    with torch.no_grad(), autocast(enabled=args.precision == 'AMP'):
        output = torch.nn.functional.softmax(model(input), dim=1)
    output = output.float().cpu().view(-1).numpy()
    top5 = np.argsort(output)[-5:][::-1]
    print(args.image)
    for (c, v) in zip(imgnet_classes[top5], output[top5]):
        print(f'{c}: {100 * v:.1f}%')

def print_in_box(msg):
    if False:
        return 10
    print('#' * (len(msg) + 10))
    print(f'#### {msg} ####')
    print('#' * (len(msg) + 10))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Classification')
    add_parser_arguments(parser)
    (args, rest) = parser.parse_known_args()
    (model_args, rest) = available_models()[args.arch].parser().parse_known_args(rest)
    assert len(rest) == 0, f'Unknown args passed: {rest}'
    cudnn.benchmark = True
    main(args, model_args)