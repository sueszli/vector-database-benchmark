import imgaug
# import imgaug.augmenters
# from imgaug import augmenters as imgaug.augmenters
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import os

# imgs = cv2.imread('1.jpg')
# print(len(imgs), '\t', len(imgs[0]))

often = lambda  aug: imgaug.augmenters.Sometimes(0.75, aug)
sometimes = lambda aug: imgaug.augmenters.Sometimes(0.5, aug)
rarely = lambda aug: imgaug.augmenters.Sometimes(0.25, aug)

"""seq = imgaug.augmenters.Sequential([
    imgaug.augmenters.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
    imgaug.augmenters.Fliplr(0.5), # 0.5 is the probability, horizontally flip 50% of the images
    imgaug.augmenters.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
])"""

"""for batch_idx in range(1000):
    # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
    # or a list of 3D numpy arrays, each having shape (height, width, channels).
    # Grayscale images must have shape (height, width, 1) each.
    # All images must have numpy's dtype uint8. Values are expected to be in
    # range 0-255.
    images = load_batch(batch_idx)
    images_aug = seq.augment_images(images)
    train_on_images(images_aug)
"""

seq = imgaug.augmenters.Sequential([
    imgaug.augmenters.Fliplr(0.5),
    imgaug.augmenters.Flipud(0.5),
    often(
        imgaug.augmenters.Affine(
            scale=(0.9,1.1),
            translate_percent=(0.05,0.1),
            rotate=(-10,10),
            shear=(-5,5),
            order=1,
            cval=0,
        )
    ),

    imgaug.augmenters.SomeOf((0,5),
    [
        rarely(
            imgaug.augmenters.Superpixels(
                p_replace=(0,1.0),
                n_segments=(20,200)
            )
        ),

        imgaug.augmenters.OneOf([
            imgaug.augmenters.GaussianBlur(sigma=(0.0, 3.0)),
            imgaug.augmenters.AverageBlur(k=(2,4)),
            imgaug.augmenters.MedianBlur(k=(3,5))
        ]),

        imgaug.augmenters.Sharpen(alpha=(0,1.0), lightness=(0.8,1.2)),

        imgaug.augmenters.Emboss(alpha=(0,1.0), strength=(0, 2.0)),
    rarely(imgaug.augmenters.OneOf([
        imgaug.augmenters.EdgeDetect(alpha=(0,0.3)),
        imgaug.augmenters.DirectedEdgeDetect(alpha=(0,0.7), direction=(0.0,1.0)),]
    )),
        imgaug.augmenters.AdditiveGaussianNoise(
            loc=0, scale=(0.0, 0.05*255)
        ),
        rarely(
            imgaug.augmenters.OneOf([
                imgaug.augmenters.Dropout((0.0,0.05),per_channel=0.5),
                imgaug.augmenters.CoarseDropout(
                    (0.03,0.05),size_percent=(0.01,0.05),per_channel=0.2
                )
            ])
        ),
        rarely(imgaug.augmenters.Invert(0.05, per_channel=True)),

        imgaug.augmenters.OneOf([
            imgaug.augmenters.Add((-10,10),per_channel=0.5),
            imgaug.augmenters.AddElementwise((-40,40))
        ]),

        imgaug.augmenters.OneOf([
            imgaug.augmenters.Multiply(mul=0.9),
            imgaug.augmenters.Multiply(mul=1.1),
        ]),

        imgaug.augmenters.Grayscale(alpha=(0.0,1.0)),

        imgaug.augmenters.ContrastNormalization((0.5,2.0)),
    ],
    random_order=True
),
    imgaug.augmenters.AddToHueAndSaturation(value=(-10,10), per_channel=True)

], random_order=True)

path = "G:/Deeplearn/signal_and_system/renametrain1"
savepath = 'G:/Deeplearn/signal_and_system/imgaug'

imglist = []
filelist = os.listdir(path)
#print(path)
count=0
for item in filelist:
    fpath = os.path.join(path, item)
    img = cv2.imread(fpath)
    #print(path+item)
    print(path+item)
    print(fpath)
    #img=np.array(img)
    print(img)
    #count+=1
    imglist.append(img)
    #if(count>=2):
        #break
#
print("append finished")
#print(imglist.shape)
#imglist=np.array(imglist)
#print(imglist.shape)
#seq.augment_images(np.array(imglist))
for count in range(100):
    images_aug = seq.augment_images(imglist)
    for index in range(len(images_aug)):
        filename = str(count)+str(index) + '.jpg'
        cv2.imwrite(savepath+filename,images_aug[index])
        print('image of count%s index%s has been writen' % (count, index))

