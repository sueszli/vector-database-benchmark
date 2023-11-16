import os, glob
import random, csv
import tensorflow as tf

def load_csv(root, filename, name2label):
    if False:
        while True:
            i = 10
    if not os.path.exists(os.path.join(root, filename)):
        images = []
        for name in name2label.keys():
            images += glob.glob(os.path.join(root, name, '*.png'))
            images += glob.glob(os.path.join(root, name, '*.jpg'))
            images += glob.glob(os.path.join(root, name, '*.jpeg'))
        print(len(images), images)
        random.shuffle(images)
        with open(os.path.join(root, filename), mode='w', newline='') as f:
            writer = csv.writer(f)
            for img in images:
                name = img.split(os.sep)[-2]
                label = name2label[name]
                writer.writerow([img, label])
            print('written into csv file:', filename)
    (images, labels) = ([], [])
    with open(os.path.join(root, filename)) as f:
        reader = csv.reader(f)
        for row in reader:
            (img, label) = row
            label = int(label)
            images.append(img)
            labels.append(label)
    return (images, labels)

def load_pokemon(root, mode='train'):
    if False:
        return 10
    name2label = {}
    for name in sorted(os.listdir(os.path.join(root))):
        if not os.path.isdir(os.path.join(root, name)):
            continue
        name2label[name] = len(name2label.keys())
    (images, labels) = load_csv(root, 'images.csv', name2label)
    if mode == 'train':
        images = images[:int(0.6 * len(images))]
        labels = labels[:int(0.6 * len(labels))]
    elif mode == 'val':
        images = images[int(0.6 * len(images)):int(0.8 * len(images))]
        labels = labels[int(0.6 * len(labels)):int(0.8 * len(labels))]
    else:
        images = images[int(0.8 * len(images)):]
        labels = labels[int(0.8 * len(labels)):]
    return (images, labels, name2label)
img_mean = tf.constant([0.485, 0.456, 0.406])
img_std = tf.constant([0.229, 0.224, 0.225])

def normalize(x, mean=img_mean, std=img_std):
    if False:
        while True:
            i = 10
    x = (x - mean) / std
    return x

def denormalize(x, mean=img_mean, std=img_std):
    if False:
        i = 10
        return i + 15
    x = x * std + mean
    return x

def preprocess(x, y):
    if False:
        return 10
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3)
    x = tf.image.resize(x, [244, 244])
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_crop(x, [224, 224, 3])
    x = tf.cast(x, dtype=tf.float32) / 255.0
    x = normalize(x)
    y = tf.convert_to_tensor(y)
    return (x, y)

def main():
    if False:
        return 10
    import time
    (images, labels, table) = load_pokemon('pokemon', 'train')
    print('images:', len(images), images)
    print('labels:', len(labels), labels)
    print('table:', table)
    db = tf.data.Dataset.from_tensor_slices((images, labels))
    db = db.shuffle(1000).map(preprocess).batch(32)
    writter = tf.summary.create_file_writer('logs')
    for (step, (x, y)) in enumerate(db):
        with writter.as_default():
            x = denormalize(x)
            tf.summary.image('img', x, step=step, max_outputs=9)
            time.sleep(5)
if __name__ == '__main__':
    main()