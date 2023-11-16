"""
Created on Thu Oct 21 11:09:09 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import copy
import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map
import torch
from torch.autograd import Variable
from torchvision import models

def convert_to_grayscale(im_as_arr):
    if False:
        i = 10
        return i + 15
    '\n        Converts 3d image to grayscale\n    Args:\n        im_as_arr (numpy arr): RGB image with shape (D,W,H)\n    returns:\n        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)\n    '
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1)
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im

def save_gradient_images(gradient, file_name):
    if False:
        for i in range(10):
            print('nop')
    '\n        Exports the original gradient image\n    Args:\n        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)\n        file_name (str): File name to be exported\n    '
    if not os.path.exists('../results'):
        os.makedirs('../results')
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    path_to_file = os.path.join('../results', file_name + '.jpg')
    save_image(gradient, path_to_file)

def save_class_activation_images(org_img, activation_map, file_name):
    if False:
        return 10
    '\n        Saves cam activation map and activation map on the original image\n    Args:\n        org_img (PIL img): Original image\n        activation_map (numpy arr): Activation map (grayscale) 0-255\n        file_name (str): File name of the exported image\n    '
    if not os.path.exists('../results'):
        os.makedirs('../results')
    (heatmap, heatmap_on_image) = apply_colormap_on_image(org_img, activation_map, 'hsv')
    path_to_file = os.path.join('../results', file_name + '_Cam_Heatmap.png')
    save_image(heatmap, path_to_file)
    path_to_file = os.path.join('../results', file_name + '_Cam_On_Image.png')
    save_image(heatmap_on_image, path_to_file)
    path_to_file = os.path.join('../results', file_name + '_Cam_Grayscale.png')
    save_image(activation_map, path_to_file)

def apply_colormap_on_image(org_im, activation, colormap_name):
    if False:
        i = 10
        return i + 15
    '\n        Apply heatmap on image\n    Args:\n        org_img (PIL img): Original image\n        activation_map (numpy arr): Activation map (grayscale) 0-255\n        colormap_name (str): Name of the colormap\n    '
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap * 255).astype(np.uint8))
    heatmap_on_image = Image.new('RGBA', org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return (no_trans_heatmap, heatmap_on_image)

def format_np_output(np_arr):
    if False:
        return 10
    '\n        This is a (kind of) bandaid fix to streamline saving procedure.\n        It converts all the outputs to the same format which is 3xWxH\n        with using sucecssive if clauses.\n    Args:\n        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH\n    '
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    if np.max(np_arr) <= 1:
        np_arr = (np_arr * 255).astype(np.uint8)
    return np_arr

def save_image(im, path):
    if False:
        i = 10
        return i + 15
    '\n        Saves a numpy matrix or PIL image as an image\n    Args:\n        im_as_arr (Numpy array): Matrix of shape DxWxH\n        path (str): Path to the image\n    '
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)

def preprocess_image(pil_im, resize_im=True):
    if False:
        i = 10
        return i + 15
    '\n        Processes image for CNNs\n    Args:\n        PIL_img (PIL_img): Image to process\n        resize_im (bool): Resize to 224 or not\n    returns:\n        im_as_var (torch variable): Variable that contains processed float tensor\n    '
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if resize_im:
        pil_im.thumbnail((224, 224))
    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)
    for (channel, _) in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    im_as_ten = torch.from_numpy(im_as_arr).float()
    im_as_ten.unsqueeze_(0)
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var

def recreate_image(im_as_var):
    if False:
        print('Hello World!')
    '\n        Recreates images from a torch variable, sort of reverse preprocessing\n    Args:\n        im_as_var (torch variable): Image to recreate\n    returns:\n        recreated_im (numpy arr): Recreated image in array\n    '
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1 / 0.229, 1 / 0.224, 1 / 0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)
    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im

def get_positive_negative_saliency(gradient):
    if False:
        i = 10
        return i + 15
    '\n        Generates positive and negative saliency maps based on the gradient\n    Args:\n        gradient (numpy arr): Gradient of the operation to visualize\n    returns:\n        pos_saliency ( )\n    '
    pos_saliency = np.maximum(0, gradient) / gradient.max()
    neg_saliency = np.maximum(0, -gradient) / -gradient.min()
    return (pos_saliency, neg_saliency)

def get_example_params(example_index):
    if False:
        return 10
    '\n        Gets used variables for almost all visualizations, like the image, model etc.\n    Args:\n        example_index (int): Image id to use from examples\n    returns:\n        original_image (numpy arr): Original image read from the file\n        prep_img (numpy_arr): Processed image\n        target_class (int): Target class for the image\n        file_name_to_export (string): File name to export the visualizations\n        pretrained_model(Pytorch model): Model to use for the operations\n    '
    example_list = (('../input_images/snake.jpg', 56), ('../input_images/cat_dog.png', 243), ('../input_images/spider.png', 72))
    img_path = example_list[example_index][0]
    target_class = example_list[example_index][1]
    file_name_to_export = img_path[img_path.rfind('/') + 1:img_path.rfind('.')]
    original_image = Image.open(img_path).convert('RGB')
    prep_img = preprocess_image(original_image)
    pretrained_model = models.alexnet(pretrained=True)
    return (original_image, prep_img, target_class, file_name_to_export, pretrained_model)