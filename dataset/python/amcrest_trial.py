import face_recognition as fr
#import skimage
from scipy.misc import imresize
import os
from face import Face
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

IMAGE_BASE_PATH = '/Users/saket/Documents/GitHub/Attendance-system/Code/session_images/21-08-2018_14h04m'
#IMAGE_BASE_PATH = '/home/deeplearning/Desktop/Attendance-System/frames/'
IMAGE = 'DSC_0236.jpg'
IMAGE_PATH = IMAGE_BASE_PATH + IMAGE

def getImages(image_dir):
    '''
    takes in image_dir as input

    returns a list of tuple:
        (image_id, numpy array of image)
    '''
    images = []
    img_names = os.listdir(image_dir)
    if(image_dir[-1] == '/'):
        IMAGE_BASE_DIR = image_dir
    else:
        IMAGE_BASE_DIR = image_dir + '/'
    for name in img_names:
	if name[0]=='.':
		continue
        img_path = IMAGE_BASE_DIR + name
        img = fr.load_image_file(img_path)
        img = cv2.resize(img, (0,0), fx = 0.7, fy = 0.7)
        img_name = name
        img_tuple = (img_name, img)
        images.append(img_tuple)
    return images

def getFaces(images, n = -1):
        '''
        images :: tuple(name, numpy array)
        n :: ground_truth number of faces in an image
        '''

        faces = []
        possible_people = []
        for img in images:
            print("Working on new image")
            fls = fr.face_locations(img[1], model = 'hog')
            encs = fr.face_encodings(img[1], fls)
            if len(fls)==0:
                print("GOT NO FACES IN IMAGE")
            if(len(fls) == n):
                start = len(faces)
                end = start + n
                possible_people.append(list(range(start,end)))
            print(len(encs), len(fls))
            b = 1
            fig,ax = plt.subplots(1)
            ax.imshow(img[1])
            for (fl, enc) in zip(fls, encs):
                face = Face()
                face.bound_box = fl
                face.img_name = img[0]
                face.box_no = b
                b += 1
                face.enc = enc
                faces.append(face)
                tlx, tly, brx, bry = fl
                #x1, y2, x2, y1
                x1 = tlx
                x2 = brx
                y1 = bry
                y2 = tly
                face.myFace = img[1][x1:x2,y1:y2]
                #print(x1,y1,x2,y2)
                if len(fls)>-1:
                    cv2.rectangle(img[1], (y1,x1), (y2,x2), (0, 0, 255), 5)
                rect = patches.Rectangle((x1,y1),(x2-x1),(y2-y1),linewidth=2,edgecolor='b')
                #ax.add_patch(rect)
            if len(fls)>-1:
                y = cv2.resize(np.array(img[1]),(0,0),fx=0.4,fy=0.4)
		cv2.imshow('disp',y)
		cv2.waitKey(1)
                #plt.show()
            #cv2.imshow(img[0], img[1])
            #cv2.waitKey()
            #cv2.destroyAllWindows()
        if(possible_people):
            return faces, possible_people[0]
        else:
            print("NO IMAGE WITH ALL PHOTOS")
            #exit(0)
            return faces, possible_people

def get_similarity(fid1,fid2,faces):
    f1,f2 = faces[fid1],faces[fid2]
    if f1.img_name==f2.img_name:
        return 0
    return (1- fr.face_distance(np.array(f1.enc.reshape(1, 128)),np.array(f2.enc.reshape(1, 128)))[0])

def print_face(n):
    img = fr.load_image_file(IMAGE_BASE_PATH+'/'+faces[n].img_name)
    tlx, tly, brx, bry = faces[n].bound_box
    x1 = tlx
    x2 = brx
    y1 = bry
    y2 = tly
    print(x1, x2, y1, y2)
    #enc = fr.face_encodings(img[1][x1: x2, y1: y2])
    plt.imshow(img[x1: x2, y1: y2])
    plt.show()

def get_sim_mat(faces):
    sim_mat = [0 for i in range(len(faces))]
    sim_mat = [sim_mat.copy() for i in range(len(faces))]

    for i in range(len(faces)):
        for j in range(i+1,len(faces)):
            sim_mat[i][j] = sim_mat[j][i] = get_similarity(i,j,faces)

    return sim_mat

images = getImages(IMAGE_BASE_PATH)
faces, people = getFaces(images,16)
#sim_mat = get_sim_mat(faces)
'''
print()
for i in range(len(sim_mat)):
    print(sim_mat[i])
'''
