import numpy as np

def tile_images(img_nhwc):
    if False:
        print('Hello World!')
    '\n    Tile N images into one big PxQ image\n    (P,Q) are chosen to be as close as possible, and if N\n    is square, then P=Q.\n\n    :param img_nhwc: (list) list or array of images, ndim=4 once turned into array. img nhwc\n        n = batch index, h = height, w = width, c = channel\n    :return: (numpy float) img_HWc, ndim=3\n    '
    img_nhwc = np.asarray(img_nhwc)
    (n_images, height, width, n_channels) = img_nhwc.shape
    new_height = int(np.ceil(np.sqrt(n_images)))
    new_width = int(np.ceil(float(n_images) / new_height))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(n_images, new_height * new_width)])
    out_image = img_nhwc.reshape(new_height, new_width, height, width, n_channels)
    out_image = out_image.transpose(0, 2, 1, 3, 4)
    out_image = out_image.reshape(new_height * height, new_width * width, n_channels)
    return out_image