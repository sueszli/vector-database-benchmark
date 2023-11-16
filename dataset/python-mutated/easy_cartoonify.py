import cv2
import os
from pathlib import Path
image_name = input('Please enter the name of the image file that you want to process:    ')
image_directory = input('Please enter the directory that may contain the image:    ')

def find_the_image(file_name, directory_name):
    if False:
        while True:
            i = 10
    files_found = []
    for (path, subdirs, files) in os.walk(directory_name):
        for name in files:
            if file_name == name:
                file_path = os.path.join(path, name)
                files_found.append(file_path)
    print(files_found[0])
    return files_found[0]
image_path = Path(find_the_image(image_name, image_directory))
new_working_directory = image_path.parent
os.chdir(new_working_directory)
color_image = cv2.imread(find_the_image(image_name, image_directory))
cartoon_style_selection = input('This script currently has 2 sytles. Please type 1 or 2.   ')
if cartoon_style_selection == '1':
    cartoon_image_style_1 = cv2.stylization(color_image, sigma_s=150, sigma_r=0.25)
    cv2.imshow('cartoon_1', cartoon_image_style_1)
    cv2.waitKey()
    cv2.destroyAllWindows()
elif cartoon_style_selection == '2':
    cartoon_image_style_2 = cv2.stylization(color_image, sigma_s=60, sigma_r=0.5)
    cv2.imshow('cartoon_2', cartoon_image_style_2)
    cv2.waitKey()
    cv2.destroyAllWindows()
else:
    print('Invalid style selection.')