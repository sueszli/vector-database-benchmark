"""
This is an example of using the k-nearest-neighbors (KNN) algorithm for face recognition.

When should I use this example?
This example is useful when you wish to recognize a large set of known people,
and make a prediction for an unknown person in a feasible computation time.

Algorithm Description:
The knn classifier is first trained on a set of labeled (known) faces and can then predict the person
in an unknown image by finding the k most similar faces (images with closet face-features under euclidean distance)
in its training set, and performing a majority vote (possibly weighted) on their label.

For example, if k=3, and the three closest face images to the given image in the training set are one image of Biden
and two images of Obama, The result would be 'Obama'.

* This implementation uses a weighted vote, such that the votes of closer-neighbors are weighted more heavily.

Usage:

1. Prepare a set of images of the known people you want to recognize. Organize the images in a single directory
   with a sub-directory for each known person.

2. Then, call the 'train' function with the appropriate parameters. Make sure to pass in the 'model_save_path' if you
   want to save the model to disk so you can re-use the model without having to re-train it.

3. Call 'predict' and pass in your trained model to recognize the people in an unknown image.

NOTE: This example requires scikit-learn to be installed! You can install it with pip:

$ pip3 install scikit-learn

"""
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    if False:
        return 10
    '\n    Trains a k-nearest neighbors classifier for face recognition.\n\n    :param train_dir: directory that contains a sub-directory for each known person, with its name.\n\n     (View in source code to see train_dir example tree structure)\n\n     Structure:\n        <train_dir>/\n        ├── <person1>/\n        │   ├── <somename1>.jpeg\n        │   ├── <somename2>.jpeg\n        │   ├── ...\n        ├── <person2>/\n        │   ├── <somename1>.jpeg\n        │   └── <somename2>.jpeg\n        └── ...\n\n    :param model_save_path: (optional) path to save model on disk\n    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified\n    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree\n    :param verbose: verbosity of training\n    :return: returns knn classifier that was trained on the given data.\n    '
    X = []
    y = []
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)
            if len(face_bounding_boxes) != 1:
                if verbose:
                    print('Image {} not suitable for training: {}'.format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else 'Found more than one face'))
            else:
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print('Chose n_neighbors automatically:', n_neighbors)
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)
    return knn_clf

def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    if False:
        while True:
            i = 10
    "\n    Recognizes faces in given image using a trained KNN classifier\n\n    :param X_img_path: path to image to be recognized\n    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.\n    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.\n    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance\n           of mis-classifying an unknown person as a known one.\n    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].\n        For faces of unrecognized persons, the name 'unknown' will be returned.\n    "
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception('Invalid image path: {}'.format(X_img_path))
    if knn_clf is None and model_path is None:
        raise Exception('Must supply knn classifier either thourgh knn_clf or model_path')
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)
    if len(X_face_locations) == 0:
        return []
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    return [(pred, loc) if rec else ('unknown', loc) for (pred, loc, rec) in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

def show_prediction_labels_on_image(img_path, predictions):
    if False:
        return 10
    '\n    Shows the face recognition results visually.\n\n    :param img_path: path to image to be recognized\n    :param predictions: results of the predict function\n    :return:\n    '
    pil_image = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(pil_image)
    for (name, (top, right, bottom, left)) in predictions:
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
        name = name.encode('UTF-8')
        (text_width, text_height) = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
    del draw
    pil_image.show()
if __name__ == '__main__':
    print('Training KNN classifier...')
    classifier = train('knn_examples/train', model_save_path='trained_knn_model.clf', n_neighbors=2)
    print('Training complete!')
    for image_file in os.listdir('knn_examples/test'):
        full_file_path = os.path.join('knn_examples/test', image_file)
        print('Looking for faces in {}'.format(image_file))
        predictions = predict(full_file_path, model_path='trained_knn_model.clf')
        for (name, (top, right, bottom, left)) in predictions:
            print('- Found {} at ({}, {})'.format(name, left, top))
        show_prediction_labels_on_image(os.path.join('knn_examples/test', image_file), predictions)