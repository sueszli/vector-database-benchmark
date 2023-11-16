def nvidia_efficientnet(type='efficient-b0', pretrained=True, **kwargs):
    if False:
        i = 10
        return i + 15
    'Constructs a EfficientNet model.\n    For detailed information on model input and output, training recipies, inference and performance\n    visit: github.com/NVIDIA/DeepLearningExamples and/or ngc.nvidia.com\n    Args:\n        pretrained (bool, True): If True, returns a model pretrained on IMAGENET dataset.\n    '
    from .efficientnet import _ce
    return _ce(type)(pretrained=pretrained, **kwargs)

def nvidia_convnets_processing_utils():
    if False:
        i = 10
        return i + 15
    import numpy as np
    import torch
    from PIL import Image
    import torchvision.transforms as transforms
    import numpy as np
    import json
    import requests
    import validators

    class Processing:

        @staticmethod
        def prepare_input_from_uri(uri, cuda=False):
            if False:
                i = 10
                return i + 15
            img_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
            if validators.url(uri):
                img = Image.open(requests.get(uri, stream=True).raw)
            else:
                img = Image.open(uri)
            img = img_transforms(img)
            with torch.no_grad():
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                img = img.float()
                if cuda:
                    mean = mean.cuda()
                    std = std.cuda()
                    img = img.cuda()
                input = img.unsqueeze(0).sub_(mean).div_(std)
            return input

        @staticmethod
        def pick_n_best(predictions, n=5):
            if False:
                print('Hello World!')
            predictions = predictions.float().cpu().numpy()
            topN = np.argsort(-1 * predictions, axis=-1)[:, :n]
            imgnet_classes = Processing.get_imgnet_classes()
            results = []
            for (idx, case) in enumerate(topN):
                r = []
                for (c, v) in zip(imgnet_classes[case], predictions[idx, case]):
                    r.append((f'{c}', f'{100 * v:.1f}%'))
                print(f'sample {idx}: {r}')
                results.append(r)
            return results

        @staticmethod
        def get_imgnet_classes():
            if False:
                print('Hello World!')
            import os
            import json
            imgnet_classes_json = 'LOC_synset_mapping.json'
            if not os.path.exists(imgnet_classes_json):
                print('Downloading Imagenet Classes names.')
                import urllib
                urllib.request.urlretrieve('https://raw.githubusercontent.com/NVIDIA/DeepLearningExamples/master/PyTorch/Classification/ConvNets/LOC_synset_mapping.json', filename=imgnet_classes_json)
                print('Downloading finished.')
            imgnet_classes = np.array(json.load(open(imgnet_classes_json, 'r')))
            return imgnet_classes
    return Processing()