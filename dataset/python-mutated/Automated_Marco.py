import tensorflow as tf
import csv
import os
import argparse
' \nusage: \nProcesses all .jpg, .png, .bmp and .gif files found in the specified directory and its subdirectories.\n --PATH ( Path to directory of images or path to directory with subdirectory of images). e.g Path/To/Directory/\n --Model_PATH path to the tensorflow model\n'
parser = argparse.ArgumentParser(description='Crystal Detection Program')
parser.add_argument('--PATH', type=str, help='path to image directory. Recursively finds all image files in directory and  sub directories')
parser.add_argument('--MODEL_PATH', type=str, default='./savedmodel', help='the file path to the tensorflow model ')
args = vars(parser.parse_args())
PATH = args['PATH']
model_path = args['MODEL_PATH']
crystal_images = [os.path.join(dp, f) for (dp, dn, filenames) in os.walk(PATH) for f in filenames if os.path.splitext(f)[1] in ['.jpg', 'png', 'bmp', 'gif']]
size = len(crystal_images)

def load_images(file_list):
    if False:
        while True:
            i = 10
    for i in file_list:
        files = open(i, 'rb')
        yield ({'image_bytes': [files.read()]}, i)
iterator = load_images(crystal_images)
with open(PATH + 'results.csv', 'w') as csvfile:
    Writer = csv.writer(csvfile, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    predicter = tf.contrib.predictor.from_saved_model(model_path)
    dic = {}
    k = 0
    for _ in range(size):
        (data, name) = next(iterator)
        results = predicter(data)
        vals = results['scores'][0]
        classes = results['classes'][0]
        dictionary = dict(zip(classes, vals))
        print('Image path: ' + name + ' Crystal: ' + str(dictionary[b'Crystals']) + ' Other: ' + str(dictionary[b'Other']) + ' Precipitate: ' + str(dictionary[b'Precipitate']) + ' Clear: ' + str(dictionary[b'Clear']))
        Writer.writerow(['Image path: ' + name, 'Crystal: ' + str(dictionary[b'Crystals']), 'Other: ' + str(dictionary[b'Other']), 'Precipitate: ' + str(dictionary[b'Precipitate']), 'Clear: ' + str(dictionary[b'Clear'])])