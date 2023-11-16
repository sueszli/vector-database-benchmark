"""
Title: A walk through latent space with Stable Diffusion
Authors: Ian Stenbit, [fchollet](https://twitter.com/fchollet), [lukewood](https://twitter.com/luke_wood_ml)
Date created: 2022/09/28
Last modified: 2022/09/28
Description: Explore the latent manifold of Stable Diffusion.
Accelerator: GPU
"""
'\n## Overview\n\nGenerative image models learn a "latent manifold" of the visual world:\na low-dimensional vector space where each point maps to an image.\nGoing from such a point on the manifold back to a displayable image\nis called "decoding" -- in the Stable Diffusion model, this is handled by\nthe "decoder" model.\n\n![The Stable Diffusion architecture](https://i.imgur.com/2uC8rYJ.png)\n\nThis latent manifold of images is continuous and interpolative, meaning that:\n\n1. Moving a little on the manifold only changes the corresponding image a little (continuity).\n2. For any two points A and B on the manifold (i.e. any two images), it is possible\nto move from A to B via a path where each intermediate point is also on the manifold (i.e.\nis also a valid image). Intermediate points would be called "interpolations" between\nthe two starting images.\n\nStable Diffusion isn\'t just an image model, though, it\'s also a natural language model.\nIt has two latent spaces: the image representation space learned by the\nencoder used during training, and the prompt latent space\nwhich is learned using a combination of pretraining and training-time\nfine-tuning.\n\n_Latent space walking_, or _latent space exploration_, is the process of\nsampling a point in latent space and incrementally changing the latent\nrepresentation. Its most common application is generating animations\nwhere each sampled point is fed to the decoder and is stored as a\nframe in the final animation.\nFor high-quality latent representations, this produces coherent-looking\nanimations. These animations can provide insight into the feature map of the\nlatent space, and can ultimately lead to improvements in the training\nprocess. One such GIF is displayed below:\n\n![Panda to Plane](/img/examples/generative/random_walks_with_stable_diffusion/panda2plane.gif)\n\nIn this guide, we will show how to take advantage of the Stable Diffusion API\nin KerasCV to perform prompt interpolation and circular walks through\nStable Diffusion\'s visual latent manifold, as well as through\nthe text encoder\'s latent manifold.\n\nThis guide assumes the reader has a\nhigh-level understanding of Stable Diffusion.\nIf you haven\'t already, you should start\nby reading the [Stable Diffusion Tutorial](https://keras.io/guides/keras_cv/generate_images_with_stable_diffusion/).\n\nTo start, we import KerasCV and load up a Stable Diffusion model using the\noptimizations discussed in the tutorial\n[Generate images with Stable Diffusion](https://keras.io/guides/keras_cv/generate_images_with_stable_diffusion/).\nNote that if you are running with a M1 Mac GPU you should not enable mixed precision.\n'
'shell\npip install keras-cv --upgrade --quiet\n'
import keras_cv
import keras
import matplotlib.pyplot as plt
from keras import ops
import numpy as np
import math
from PIL import Image
keras.mixed_precision.set_global_policy('mixed_float16')
model = keras_cv.models.StableDiffusion(jit_compile=True)
"\n## Interpolating between text prompts\n\nIn Stable Diffusion, a text prompt is first encoded into a vector,\nand that encoding is used to guide the diffusion process.\nThe latent encoding vector has shape\n77x768 (that's huge!), and when we give Stable Diffusion a text prompt, we're\ngenerating images from just one such point on the latent manifold.\n\nTo explore more of this manifold, we can interpolate between two text encodings\nand generate images at those interpolated points:\n"
prompt_1 = 'A watercolor painting of a Golden Retriever at the beach'
prompt_2 = 'A still life DSLR photo of a bowl of fruit'
interpolation_steps = 5
encoding_1 = ops.squeeze(model.encode_text(prompt_1))
encoding_2 = ops.squeeze(model.encode_text(prompt_2))
interpolated_encodings = ops.linspace(encoding_1, encoding_2, interpolation_steps)
print(f'Encoding shape: {encoding_1.shape}')
"\nOnce we've interpolated the encodings, we can generate images from each point.\nNote that in order to maintain some stability between the resulting images we\nkeep the diffusion noise constant between images.\n"
seed = 12345
noise = keras.random.normal((512 // 8, 512 // 8, 4), seed=seed)
images = model.generate_image(interpolated_encodings, batch_size=interpolation_steps, diffusion_noise=noise)
'\nNow that we\'ve generated some interpolated images, let\'s take a look at them!\n\nThroughout this tutorial, we\'re going to export sequences of images as gifs so\nthat they can be easily viewed with some temporal context. For sequences of\nimages where the first and last images don\'t match conceptually, we rubber-band\nthe gif.\n\nIf you\'re running in Colab, you can view your own GIFs by running:\n\n```\nfrom IPython.display import Image as IImage\nIImage("doggo-and-fruit-5.gif")\n```\n'

def export_as_gif(filename, images, frames_per_second=10, rubber_band=False):
    if False:
        return 10
    if rubber_band:
        images += images[2:-1][::-1]
    images[0].save(filename, save_all=True, append_images=images[1:], duration=1000 // frames_per_second, loop=0)
export_as_gif('doggo-and-fruit-5.gif', [Image.fromarray(img) for img in images], frames_per_second=2, rubber_band=True)
"\n![Dog to Fruit 5](https://i.imgur.com/4ZCxZY4.gif)\n\nThe results may seem surprising. Generally, interpolating between prompts\nproduces coherent looking images, and often demonstrates a progressive concept\nshift between the contents of the two prompts. This is indicative of a high\nquality representation space, that closely mirrors the natural structure\nof the visual world.\n\nTo best visualize this, we should do a much more fine-grained interpolation,\nusing hundreds of steps. In order to keep batch size small (so that we don't\nOOM our GPU), this requires manually batching our interpolated\nencodings.\n"
interpolation_steps = 150
batch_size = 3
batches = interpolation_steps // batch_size
interpolated_encodings = ops.linspace(encoding_1, encoding_2, interpolation_steps)
batched_encodings = ops.split(interpolated_encodings, batches)
images = []
for batch in range(batches):
    images += [Image.fromarray(img) for img in model.generate_image(batched_encodings[batch], batch_size=batch_size, num_steps=25, diffusion_noise=noise)]
export_as_gif('doggo-and-fruit-150.gif', images, rubber_band=True)
'\n![Dog to Fruit 150](/img/examples/generative/random_walks_with_stable_diffusion/dog2fruit150.gif)\n\nThe resulting gif shows a much clearer and more coherent shift between the two\nprompts. Try out some prompts of your own and experiment!\n\nWe can even extend this concept for more than one image. For example, we can\ninterpolate between four prompts:\n'
prompt_1 = 'A watercolor painting of a Golden Retriever at the beach'
prompt_2 = 'A still life DSLR photo of a bowl of fruit'
prompt_3 = 'The eiffel tower in the style of starry night'
prompt_4 = 'An architectural sketch of a skyscraper'
interpolation_steps = 6
batch_size = 3
batches = interpolation_steps ** 2 // batch_size
encoding_1 = ops.squeeze(model.encode_text(prompt_1))
encoding_2 = ops.squeeze(model.encode_text(prompt_2))
encoding_3 = ops.squeeze(model.encode_text(prompt_3))
encoding_4 = ops.squeeze(model.encode_text(prompt_4))
interpolated_encodings = ops.linspace(ops.linspace(encoding_1, encoding_2, interpolation_steps), ops.linspace(encoding_3, encoding_4, interpolation_steps), interpolation_steps)
interpolated_encodings = ops.reshape(interpolated_encodings, (interpolation_steps ** 2, 77, 768))
batched_encodings = ops.split(interpolated_encodings, batches)
images = []
for batch in range(batches):
    images.append(model.generate_image(batched_encodings[batch], batch_size=batch_size, diffusion_noise=noise))

def plot_grid(images, path, grid_size, scale=2):
    if False:
        while True:
            i = 10
    fig = plt.figure(figsize=(grid_size * scale, grid_size * scale))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(x=0, y=0)
    plt.axis('off')
    images = images.astype(int)
    for row in range(grid_size):
        for col in range(grid_size):
            index = row * grid_size + col
            plt.subplot(grid_size, grid_size, index + 1)
            plt.imshow(images[index].astype('uint8'))
            plt.axis('off')
            plt.margins(x=0, y=0)
    plt.savefig(fname=path, pad_inches=0, bbox_inches='tight', transparent=False, dpi=60)
images = np.concatenate(images)
plot_grid(images, '4-way-interpolation.jpg', interpolation_steps)
'\nWe can also interpolate while allowing diffusion noise to vary by dropping\nthe `diffusion_noise` parameter:\n'
images = []
for batch in range(batches):
    images.append(model.generate_image(batched_encodings[batch], batch_size=batch_size))
images = np.concatenate(images)
plot_grid(images, '4-way-interpolation-varying-noise.jpg', interpolation_steps)
"\nNext up -- let's go for some walks!\n\n## A walk around a text prompt\n\nOur next experiment will be to go for a walk around the latent manifold\nstarting from a point produced by a particular prompt.\n"
walk_steps = 150
batch_size = 3
batches = walk_steps // batch_size
step_size = 0.005
encoding = ops.squeeze(model.encode_text('The Eiffel Tower in the style of starry night'))
delta = ops.ones_like(encoding) * step_size
walked_encodings = []
for step_index in range(walk_steps):
    walked_encodings.append(encoding)
    encoding += delta
walked_encodings = ops.stack(walked_encodings)
batched_encodings = ops.split(walked_encodings, batches)
images = []
for batch in range(batches):
    images += [Image.fromarray(img) for img in model.generate_image(batched_encodings[batch], batch_size=batch_size, num_steps=25, diffusion_noise=noise)]
export_as_gif('eiffel-tower-starry-night.gif', images, rubber_band=True)
'\n![Eiffel tower walk gif](https://i.imgur.com/9MMYtal.gif)\n\nPerhaps unsurprisingly, walking too far from the encoder\'s latent manifold\nproduces images that look incoherent. Try it for yourself by setting\nyour own prompt, and adjusting `step_size` to increase or decrease the magnitude\nof the walk. Note that when the magnitude of the walk gets large, the walk often\nleads into areas which produce extremely noisy images.\n\n## A circular walk through the diffusion noise space for a single prompt\n\nOur final experiment is to stick to one prompt and explore the variety of images\nthat the diffusion model can produce from that prompt. We do this by controlling\nthe noise that is used to seed the diffusion process.\n\nWe create two noise components, `x` and `y`, and do a walk from 0 to 2π, summing\nthe cosine of our `x` component and the sin of our `y` component to produce noise.\nUsing this approach, the end of our walk arrives at the same noise inputs where\nwe began our walk, so we get a "loopable" result!\n'
prompt = 'An oil paintings of cows in a field next to a windmill in Holland'
encoding = ops.squeeze(model.encode_text(prompt))
walk_steps = 150
batch_size = 3
batches = walk_steps // batch_size
walk_noise_x = keras.random.normal(noise.shape, dtype='float64')
walk_noise_y = keras.random.normal(noise.shape, dtype='float64')
walk_scale_x = ops.cos(ops.linspace(0, 2, walk_steps) * math.pi)
walk_scale_y = ops.sin(ops.linspace(0, 2, walk_steps) * math.pi)
noise_x = ops.tensordot(walk_scale_x, walk_noise_x, axes=0)
noise_y = ops.tensordot(walk_scale_y, walk_noise_y, axes=0)
noise = ops.add(noise_x, noise_y)
batched_noise = ops.split(noise, batches)
images = []
for batch in range(batches):
    images += [Image.fromarray(img) for img in model.generate_image(encoding, batch_size=batch_size, num_steps=25, diffusion_noise=batched_noise[batch])]
export_as_gif('cows.gif', images)
'\n![Happy Cows](/img/examples/generative/random_walks_with_stable_diffusion/happycows.gif)\n\nExperiment with your own prompts and with different values of\n`unconditional_guidance_scale`!\n\n## Conclusion\n\nStable Diffusion offers a lot more than just single text-to-image generation.\nExploring the latent manifold of the text encoder and the noise space of the\ndiffusion model are two fun ways to experience the power of this model, and\nKerasCV makes it easy!\n'