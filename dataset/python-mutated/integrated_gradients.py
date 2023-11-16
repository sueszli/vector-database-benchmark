"""
Title: Model interpretability with Integrated Gradients
Author: [A_K_Nain](https://twitter.com/A_K_Nain)
Date created: 2020/06/02
Last modified: 2020/06/02
Description: How to obtain integrated gradients for a classification model.
Accelerator: NONE
"""
"\n## Integrated Gradients\n\n[Integrated Gradients](https://arxiv.org/abs/1703.01365) is a technique for\nattributing a classification model's prediction to its input features. It is\na model interpretability technique: you can use it to visualize the relationship\nbetween input features and model predictions.\n\nIntegrated Gradients is a variation on computing\nthe gradient of the prediction output with regard to features of the input.\nTo compute integrated gradients, we need to perform the following steps:\n\n1. Identify the input and the output. In our case, the input is an image and the\noutput is the last layer of our model (dense layer with softmax activation).\n\n2. Compute which features are important to a neural network\nwhen making a prediction on a particular data point. To identify these features, we\nneed to choose a baseline input. A baseline input can be a black image (all pixel\nvalues set to zero) or random noise. The shape of the baseline input needs to be\nthe same as our input image, e.g. (299, 299, 3).\n\n3. Interpolate the baseline for a given number of steps. The number of steps represents\nthe steps we need in the gradient approximation for a given input image. The number of\nsteps is a hyperparameter. The authors recommend using anywhere between\n20 and 1000 steps.\n\n4. Preprocess these interpolated images and do a forward pass.\n5. Get the gradients for these interpolated images.\n6. Approximate the gradients integral using the trapezoidal rule.\n\nTo read in-depth about integrated gradients and why this method works,\nconsider reading this excellent\n[article](https://distill.pub/2020/attribution-baselines/).\n\n**References:**\n\n- Integrated Gradients original [paper](https://arxiv.org/abs/1703.01365)\n- [Original implementation](https://github.com/ankurtaly/Integrated-Gradients)\n"
'\n## Setup\n'
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from IPython.display import Image, display
import tensorflow as tf
import keras
from keras import layers
from keras.applications import xception
keras.config.disable_traceback_filtering()
img_size = (299, 299, 3)
model = xception.Xception(weights='imagenet')
img_path = keras.utils.get_file('elephant.jpg', 'https://i.imgur.com/Bvro0YD.png')
display(Image(img_path))
'\n## Integrated Gradients algorithm\n'

def get_img_array(img_path, size=(299, 299)):
    if False:
        print('Hello World!')
    img = keras.utils.load_img(img_path, target_size=size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def get_gradients(img_input, top_pred_idx):
    if False:
        print('Hello World!')
    'Computes the gradients of outputs w.r.t input image.\n\n    Args:\n        img_input: 4D image tensor\n        top_pred_idx: Predicted label for the input image\n\n    Returns:\n        Gradients of the predictions w.r.t img_input\n    '
    images = tf.cast(img_input, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(images)
        preds = model(images)
        top_class = preds[:, top_pred_idx]
    grads = tape.gradient(top_class, images)
    return grads

def get_integrated_gradients(img_input, top_pred_idx, baseline=None, num_steps=50):
    if False:
        return 10
    'Computes Integrated Gradients for a predicted label.\n\n    Args:\n        img_input (ndarray): Original image\n        top_pred_idx: Predicted label for the input image\n        baseline (ndarray): The baseline image to start with for interpolation\n        num_steps: Number of interpolation steps between the baseline\n            and the input used in the computation of integrated gradients. These\n            steps along determine the integral approximation error. By default,\n            num_steps is set to 50.\n\n    Returns:\n        Integrated gradients w.r.t input image\n    '
    if baseline is None:
        baseline = np.zeros(img_size).astype(np.float32)
    else:
        baseline = baseline.astype(np.float32)
    img_input = img_input.astype(np.float32)
    interpolated_image = [baseline + step / num_steps * (img_input - baseline) for step in range(num_steps + 1)]
    interpolated_image = np.array(interpolated_image).astype(np.float32)
    interpolated_image = xception.preprocess_input(interpolated_image)
    grads = []
    for (i, img) in enumerate(interpolated_image):
        img = tf.expand_dims(img, axis=0)
        grad = get_gradients(img, top_pred_idx=top_pred_idx)
        grads.append(grad[0])
    grads = tf.convert_to_tensor(grads, dtype=tf.float32)
    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = tf.reduce_mean(grads, axis=0)
    integrated_grads = (img_input - baseline) * avg_grads
    return integrated_grads

def random_baseline_integrated_gradients(img_input, top_pred_idx, num_steps=50, num_runs=2):
    if False:
        for i in range(10):
            print('nop')
    'Generates a number of random baseline images.\n\n    Args:\n        img_input (ndarray): 3D image\n        top_pred_idx: Predicted label for the input image\n        num_steps: Number of interpolation steps between the baseline\n            and the input used in the computation of integrated gradients. These\n            steps along determine the integral approximation error. By default,\n            num_steps is set to 50.\n        num_runs: number of baseline images to generate\n\n    Returns:\n        Averaged integrated gradients for `num_runs` baseline images\n    '
    integrated_grads = []
    for run in range(num_runs):
        baseline = np.random.random(img_size) * 255
        igrads = get_integrated_gradients(img_input=img_input, top_pred_idx=top_pred_idx, baseline=baseline, num_steps=num_steps)
        integrated_grads.append(igrads)
    integrated_grads = tf.convert_to_tensor(integrated_grads)
    return tf.reduce_mean(integrated_grads, axis=0)
'\n## Helper class for visualizing gradients and integrated gradients\n'

class GradVisualizer:
    """Plot gradients of the outputs w.r.t an input image."""

    def __init__(self, positive_channel=None, negative_channel=None):
        if False:
            return 10
        if positive_channel is None:
            self.positive_channel = [0, 255, 0]
        else:
            self.positive_channel = positive_channel
        if negative_channel is None:
            self.negative_channel = [255, 0, 0]
        else:
            self.negative_channel = negative_channel

    def apply_polarity(self, attributions, polarity):
        if False:
            return 10
        if polarity == 'positive':
            return np.clip(attributions, 0, 1)
        else:
            return np.clip(attributions, -1, 0)

    def apply_linear_transformation(self, attributions, clip_above_percentile=99.9, clip_below_percentile=70.0, lower_end=0.2):
        if False:
            print('Hello World!')
        m = self.get_thresholded_attributions(attributions, percentage=100 - clip_above_percentile)
        e = self.get_thresholded_attributions(attributions, percentage=100 - clip_below_percentile)
        transformed_attributions = (1 - lower_end) * (np.abs(attributions) - e) / (m - e) + lower_end
        transformed_attributions *= np.sign(attributions)
        transformed_attributions *= transformed_attributions >= lower_end
        transformed_attributions = np.clip(transformed_attributions, 0.0, 1.0)
        return transformed_attributions

    def get_thresholded_attributions(self, attributions, percentage):
        if False:
            i = 10
            return i + 15
        if percentage == 100.0:
            return np.min(attributions)
        flatten_attr = attributions.flatten()
        total = np.sum(flatten_attr)
        sorted_attributions = np.sort(np.abs(flatten_attr))[::-1]
        cum_sum = 100.0 * np.cumsum(sorted_attributions) / total
        indices_to_consider = np.where(cum_sum >= percentage)[0][0]
        attributions = sorted_attributions[indices_to_consider]
        return attributions

    def binarize(self, attributions, threshold=0.001):
        if False:
            return 10
        return attributions > threshold

    def morphological_cleanup_fn(self, attributions, structure=np.ones((4, 4))):
        if False:
            print('Hello World!')
        closed = ndimage.grey_closing(attributions, structure=structure)
        opened = ndimage.grey_opening(closed, structure=structure)
        return opened

    def draw_outlines(self, attributions, percentage=90, connected_component_structure=np.ones((3, 3))):
        if False:
            while True:
                i = 10
        attributions = self.binarize(attributions)
        attributions = ndimage.binary_fill_holes(attributions)
        (connected_components, num_comp) = ndimage.label(attributions, structure=connected_component_structure)
        total = np.sum(attributions[connected_components > 0])
        component_sums = []
        for comp in range(1, num_comp + 1):
            mask = connected_components == comp
            component_sum = np.sum(attributions[mask])
            component_sums.append((component_sum, mask))
        sorted_sums_and_masks = sorted(component_sums, key=lambda x: x[0], reverse=True)
        sorted_sums = list(zip(*sorted_sums_and_masks))[0]
        cumulative_sorted_sums = np.cumsum(sorted_sums)
        cutoff_threshold = percentage * total / 100
        cutoff_idx = np.where(cumulative_sorted_sums >= cutoff_threshold)[0][0]
        if cutoff_idx > 2:
            cutoff_idx = 2
        border_mask = np.zeros_like(attributions)
        for i in range(cutoff_idx + 1):
            border_mask[sorted_sums_and_masks[i][1]] = 1
        eroded_mask = ndimage.binary_erosion(border_mask, iterations=1)
        border_mask[eroded_mask] = 0
        return border_mask

    def process_grads(self, image, attributions, polarity='positive', clip_above_percentile=99.9, clip_below_percentile=0, morphological_cleanup=False, structure=np.ones((3, 3)), outlines=False, outlines_component_percentage=90, overlay=True):
        if False:
            i = 10
            return i + 15
        if polarity not in ['positive', 'negative']:
            raise ValueError(f" Allowed polarity values: 'positive' or 'negative'\n                                    but provided {polarity}")
        if clip_above_percentile < 0 or clip_above_percentile > 100:
            raise ValueError('clip_above_percentile must be in [0, 100]')
        if clip_below_percentile < 0 or clip_below_percentile > 100:
            raise ValueError('clip_below_percentile must be in [0, 100]')
        if polarity == 'positive':
            attributions = self.apply_polarity(attributions, polarity=polarity)
            channel = self.positive_channel
        else:
            attributions = self.apply_polarity(attributions, polarity=polarity)
            attributions = np.abs(attributions)
            channel = self.negative_channel
        attributions = np.average(attributions, axis=2)
        attributions = self.apply_linear_transformation(attributions, clip_above_percentile=clip_above_percentile, clip_below_percentile=clip_below_percentile, lower_end=0.0)
        if morphological_cleanup:
            attributions = self.morphological_cleanup_fn(attributions, structure=structure)
        if outlines:
            attributions = self.draw_outlines(attributions, percentage=outlines_component_percentage)
        attributions = np.expand_dims(attributions, 2) * channel
        if overlay:
            attributions = np.clip(attributions * 0.8 + image, 0, 255)
        return attributions

    def visualize(self, image, gradients, integrated_gradients, polarity='positive', clip_above_percentile=99.9, clip_below_percentile=0, morphological_cleanup=False, structure=np.ones((3, 3)), outlines=False, outlines_component_percentage=90, overlay=True, figsize=(15, 8)):
        if False:
            for i in range(10):
                print('nop')
        img1 = np.copy(image)
        img2 = np.copy(image)
        grads_attr = self.process_grads(image=img1, attributions=gradients, polarity=polarity, clip_above_percentile=clip_above_percentile, clip_below_percentile=clip_below_percentile, morphological_cleanup=morphological_cleanup, structure=structure, outlines=outlines, outlines_component_percentage=outlines_component_percentage, overlay=overlay)
        igrads_attr = self.process_grads(image=img2, attributions=integrated_gradients, polarity=polarity, clip_above_percentile=clip_above_percentile, clip_below_percentile=clip_below_percentile, morphological_cleanup=morphological_cleanup, structure=structure, outlines=outlines, outlines_component_percentage=outlines_component_percentage, overlay=overlay)
        (_, ax) = plt.subplots(1, 3, figsize=figsize)
        ax[0].imshow(image)
        ax[1].imshow(grads_attr.astype(np.uint8))
        ax[2].imshow(igrads_attr.astype(np.uint8))
        ax[0].set_title('Input')
        ax[1].set_title('Normal gradients')
        ax[2].set_title('Integrated gradients')
        plt.show()
"\n## Let's test-drive it\n"
img = get_img_array(img_path)
orig_img = np.copy(img[0]).astype(np.uint8)
img_processed = tf.cast(xception.preprocess_input(img), dtype=tf.float32)
preds = model.predict(img_processed)
top_pred_idx = tf.argmax(preds[0])
print('Predicted:', top_pred_idx, xception.decode_predictions(preds, top=1)[0])
grads = get_gradients(img_processed, top_pred_idx=top_pred_idx)
igrads = random_baseline_integrated_gradients(np.copy(orig_img), top_pred_idx=top_pred_idx, num_steps=50, num_runs=2)
vis = GradVisualizer()
vis.visualize(image=orig_img, gradients=grads[0].numpy(), integrated_gradients=igrads.numpy(), clip_above_percentile=99, clip_below_percentile=0)
vis.visualize(image=orig_img, gradients=grads[0].numpy(), integrated_gradients=igrads.numpy(), clip_above_percentile=95, clip_below_percentile=28, morphological_cleanup=True, outlines=True)