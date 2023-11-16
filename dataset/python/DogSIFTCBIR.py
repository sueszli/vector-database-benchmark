import cv2 as cv
import os
import argparse
from collections import defaultdict
import numpy as np
from os import listdir
from os import path
import matplotlib.pyplot as plt
from skimage.io import imread


def feature_matcher(query_image, image_folder, method="surf", top_n=5):
    matches_scores = defaultdict()

    # Generate the image descriptor
    img1 = cv.imread(query_image, 0)
    cv_descriptor = cv.xfeatures2d.SIFT_create(nfeatures=800)
    # else:
    #     cv_descriptor = cv.xfeatures2d.SURF_create(800)
    kp1, des1 = cv_descriptor.detectAndCompute(img1, None)

    # Generate the Matcher object, in this case the Brute Force Matcher
    bf = cv.BFMatcher(cv.NORM_L2)


    count = 0
    if path.isdir(image_folder):
        print(image_folder)
        dirs = listdir(image_folder)
        for dir in dirs:
            dir_path = path.join(image_folder, dir)
            if path.isdir(dir_path):
                print(dir_path)
                images = listdir(image_folder+'/'+dir)
                for img in images:
                    # if count == 100:
                    #     break
                    try:
                        train_image = image_folder + '/' + dir + '/' + img
                        img2 = cv.imread(train_image, 0)
                        # Generate the descriptors for the image to be compared
                        surf = cv.xfeatures2d.SIFT_create(800)
                        kp2, des2 = surf.detectAndCompute(img2, None)

                        # Perform knnMatch to retrieve the number of matches
                        matches = bf.knnMatch(des1, des2, k=2)

                        # Filter the matches and retain only the good matches
                        good = []
                        for m, n in matches:
                            if m.distance < 0.7 * n.distance:
                                good.append(m)

                        # Store the number of good match
                        matches_scores[train_image] = len(good)

                    except:
                        pass
                    count += 1

    # Return the images with the best number of matches, we do not need to
    # divide by total since it is a fixed total value
    return dict(sorted(matches_scores.items(), key=lambda x: x[1],
                       reverse=True)[:top_n]).keys()


if __name__ == '__main__':
    # parse cmd args
    # parser = argparse.ArgumentParser(
    #     description="Feature Matching Image Retrieval"
    # )
    # parser.add_argument('--query_image', dest="query", required=True)
    # parser.add_argument('--image_dir', dest="image_dir", required=True)
    # parser.add_argument('--top_n', dest="top_n",
    #                     default=5, type=int)
    #
    # args = vars(parser.parse_args())
    # print(args)
    train_src = 'dogImages/train'
    test_ids = []
    with open('test_data.csv', 'r') as csvFile:
        data = csvFile.readlines()
    for row in data:
        vals = row.split(',')
        f = [float(x.strip()) for x in vals[1:]]
        test_ids.append((vals[0]))

    index = np.random.choice(len(test_ids))
    query = test_ids[index]

    top_match = feature_matcher(query, train_src)
    print("Top matches are:\n")
    # print(top_match)
    res =[]
    for key in top_match:
        res.append(key)
    print(res)

    fig, axes1 = plt.subplots(2, 2, figsize=(5, 5))
    i = 0
    for j in range(2):
        for k in range(2):
            if j == 0 and k == 0:
                # axes1[0][0].set_axis_off()
                image= imread(query)
                axes1[0][0].imshow(image)
                axes1[0][0].title.set_text('Query Image')
                axes1[0][0].spines['bottom'].set_color('red')
                axes1[0][0].spines['top'].set_color('red')
                axes1[0][0].spines['right'].set_color('red')
                axes1[0][0].spines['left'].set_color('red')
            else:
                # print(id)
                axes1[j][k].set_axis_off()
                image = imread(res[i])
                axes1[j][k].imshow(image)
                i += 1

    plt.show()
