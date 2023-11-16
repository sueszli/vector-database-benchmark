import os
from os import path
import warnings
import time
import pickle
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import tensorflow as tf
from deprecated import deprecated
from deepface.basemodels import VGGFace, OpenFace, Facenet, Facenet512, FbDeepFace, DeepID, DlibWrapper, ArcFace, SFace
from deepface.extendedmodels import Age, Gender, Race, Emotion
from deepface.commons import functions, realtime, distance as dst
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf_version = int(tf.__version__.split('.', maxsplit=1)[0])
if tf_version == 2:
    tf.get_logger().setLevel(logging.ERROR)

def build_model(model_name):
    if False:
        for i in range(10):
            print('nop')
    '\n    This function builds a deepface model\n    Parameters:\n            model_name (string): face recognition or facial attribute model\n                    VGG-Face, Facenet, OpenFace, DeepFace, DeepID for face recognition\n                    Age, Gender, Emotion, Race for facial attributes\n\n    Returns:\n            built deepface model\n    '
    global model_obj
    models = {'VGG-Face': VGGFace.loadModel, 'OpenFace': OpenFace.loadModel, 'Facenet': Facenet.loadModel, 'Facenet512': Facenet512.loadModel, 'DeepFace': FbDeepFace.loadModel, 'DeepID': DeepID.loadModel, 'Dlib': DlibWrapper.loadModel, 'ArcFace': ArcFace.loadModel, 'SFace': SFace.load_model, 'Emotion': Emotion.loadModel, 'Age': Age.loadModel, 'Gender': Gender.loadModel, 'Race': Race.loadModel}
    if not 'model_obj' in globals():
        model_obj = {}
    if not model_name in model_obj:
        model = models.get(model_name)
        if model:
            model = model()
            model_obj[model_name] = model
        else:
            raise ValueError(f'Invalid model_name passed - {model_name}')
    return model_obj[model_name]

def verify(img1_path, img2_path, model_name='VGG-Face', detector_backend='opencv', distance_metric='cosine', enforce_detection=True, align=True, normalization='base'):
    if False:
        i = 10
        return i + 15
    '\n    This function verifies an image pair is same person or different persons. In the background,\n    verification function represents facial images as vectors and then calculates the similarity\n    between those vectors. Vectors of same person images should have more similarity (or less\n    distance) than vectors of different persons.\n\n    Parameters:\n            img1_path, img2_path: exact image path as string. numpy array (BGR) or based64 encoded\n            images are also welcome. If one of pair has more than one face, then we will compare the\n            face pair with max similarity.\n\n            model_name (str): VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib\n            , ArcFace and SFace\n\n            distance_metric (string): cosine, euclidean, euclidean_l2\n\n            enforce_detection (boolean): If no face could not be detected in an image, then this\n            function will return exception by default. Set this to False not to have this exception.\n            This might be convenient for low resolution images.\n\n            detector_backend (string): set face detector backend to opencv, retinaface, mtcnn, ssd,\n            dlib, mediapipe or yolov8.\n\n            align (boolean): alignment according to the eye positions.\n\n            normalization (string): normalize the input image before feeding to model\n\n    Returns:\n            Verify function returns a dictionary.\n\n            {\n                    "verified": True\n                    , "distance": 0.2563\n                    , "max_threshold_to_verify": 0.40\n                    , "model": "VGG-Face"\n                    , "similarity_metric": "cosine"\n                    , \'facial_areas\': {\n                            \'img1\': {\'x\': 345, \'y\': 211, \'w\': 769, \'h\': 769},\n                            \'img2\': {\'x\': 318, \'y\': 534, \'w\': 779, \'h\': 779}\n                    }\n                    , "time": 2\n            }\n\n    '
    tic = time.time()
    target_size = functions.find_target_size(model_name=model_name)
    img1_objs = functions.extract_faces(img=img1_path, target_size=target_size, detector_backend=detector_backend, grayscale=False, enforce_detection=enforce_detection, align=align)
    img2_objs = functions.extract_faces(img=img2_path, target_size=target_size, detector_backend=detector_backend, grayscale=False, enforce_detection=enforce_detection, align=align)
    distances = []
    regions = []
    for (img1_content, img1_region, _) in img1_objs:
        for (img2_content, img2_region, _) in img2_objs:
            img1_embedding_obj = represent(img_path=img1_content, model_name=model_name, enforce_detection=enforce_detection, detector_backend='skip', align=align, normalization=normalization)
            img2_embedding_obj = represent(img_path=img2_content, model_name=model_name, enforce_detection=enforce_detection, detector_backend='skip', align=align, normalization=normalization)
            img1_representation = img1_embedding_obj[0]['embedding']
            img2_representation = img2_embedding_obj[0]['embedding']
            if distance_metric == 'cosine':
                distance = dst.findCosineDistance(img1_representation, img2_representation)
            elif distance_metric == 'euclidean':
                distance = dst.findEuclideanDistance(img1_representation, img2_representation)
            elif distance_metric == 'euclidean_l2':
                distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation))
            else:
                raise ValueError('Invalid distance_metric passed - ', distance_metric)
            distances.append(distance)
            regions.append((img1_region, img2_region))
    threshold = dst.findThreshold(model_name, distance_metric)
    distance = min(distances)
    facial_areas = regions[np.argmin(distances)]
    toc = time.time()
    resp_obj = {'verified': distance <= threshold, 'distance': distance, 'threshold': threshold, 'model': model_name, 'detector_backend': detector_backend, 'similarity_metric': distance_metric, 'facial_areas': {'img1': facial_areas[0], 'img2': facial_areas[1]}, 'time': round(toc - tic, 2)}
    return resp_obj

def analyze(img_path, actions=('emotion', 'age', 'gender', 'race'), enforce_detection=True, detector_backend='opencv', align=True, silent=False):
    if False:
        return 10
    '\n    This function analyzes facial attributes including age, gender, emotion and race.\n    In the background, analysis function builds convolutional neural network models to\n    classify age, gender, emotion and race of the input image.\n\n    Parameters:\n            img_path: exact image path, numpy array (BGR) or base64 encoded image could be passed.\n            If source image has more than one face, then result will be size of number of faces\n            appearing in the image.\n\n            actions (tuple): The default is (\'age\', \'gender\', \'emotion\', \'race\'). You can drop\n            some of those attributes.\n\n            enforce_detection (bool): The function throws exception if no face detected by default.\n            Set this to False if you don\'t want to get exception. This might be convenient for low\n            resolution images.\n\n            detector_backend (string): set face detector backend to opencv, retinaface, mtcnn, ssd,\n            dlib, mediapipe or yolov8.\n\n            align (boolean): alignment according to the eye positions.\n\n            silent (boolean): disable (some) log messages\n\n    Returns:\n            The function returns a list of dictionaries for each face appearing in the image.\n\n            [\n                    {\n                            "region": {\'x\': 230, \'y\': 120, \'w\': 36, \'h\': 45},\n                            "age": 28.66,\n                            \'face_confidence\': 0.9993908405303955,\n                            "dominant_gender": "Woman",\n                            "gender": {\n                                    \'Woman\': 99.99407529830933,\n                                    \'Man\': 0.005928758764639497,\n                            }\n                            "dominant_emotion": "neutral",\n                            "emotion": {\n                                    \'sad\': 37.65260875225067,\n                                    \'angry\': 0.15512987738475204,\n                                    \'surprise\': 0.0022171278033056296,\n                                    \'fear\': 1.2489334680140018,\n                                    \'happy\': 4.609785228967667,\n                                    \'disgust\': 9.698561953541684e-07,\n                                    \'neutral\': 56.33133053779602\n                            }\n                            "dominant_race": "white",\n                            "race": {\n                                    \'indian\': 0.5480832420289516,\n                                    \'asian\': 0.7830780930817127,\n                                    \'latino hispanic\': 2.0677512511610985,\n                                    \'black\': 0.06337375962175429,\n                                    \'middle eastern\': 3.088453598320484,\n                                    \'white\': 93.44925880432129\n                            }\n                    }\n            ]\n    '
    if isinstance(actions, str):
        actions = (actions,)
    if not hasattr(actions, '__getitem__') or not actions:
        raise ValueError('`actions` must be a list of strings.')
    actions = list(actions)
    for action in actions:
        if action not in ('emotion', 'age', 'gender', 'race'):
            raise ValueError(f'Invalid action passed ({repr(action)})). Valid actions are `emotion`, `age`, `gender`, `race`.')
    models = {}
    if 'emotion' in actions:
        models['emotion'] = build_model('Emotion')
    if 'age' in actions:
        models['age'] = build_model('Age')
    if 'gender' in actions:
        models['gender'] = build_model('Gender')
    if 'race' in actions:
        models['race'] = build_model('Race')
    resp_objects = []
    img_objs = functions.extract_faces(img=img_path, target_size=(224, 224), detector_backend=detector_backend, grayscale=False, enforce_detection=enforce_detection, align=align)
    for (img_content, img_region, img_confidence) in img_objs:
        if img_content.shape[0] > 0 and img_content.shape[1] > 0:
            obj = {}
            pbar = tqdm(range(0, len(actions)), desc='Finding actions', disable=silent)
            for index in pbar:
                action = actions[index]
                pbar.set_description(f'Action: {action}')
                if action == 'emotion':
                    img_gray = cv2.cvtColor(img_content[0], cv2.COLOR_BGR2GRAY)
                    img_gray = cv2.resize(img_gray, (48, 48))
                    img_gray = np.expand_dims(img_gray, axis=0)
                    emotion_predictions = models['emotion'].predict(img_gray, verbose=0)[0, :]
                    sum_of_predictions = emotion_predictions.sum()
                    obj['emotion'] = {}
                    for (i, emotion_label) in enumerate(Emotion.labels):
                        emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
                        obj['emotion'][emotion_label] = emotion_prediction
                    obj['dominant_emotion'] = Emotion.labels[np.argmax(emotion_predictions)]
                elif action == 'age':
                    age_predictions = models['age'].predict(img_content, verbose=0)[0, :]
                    apparent_age = Age.findApparentAge(age_predictions)
                    obj['age'] = int(apparent_age)
                elif action == 'gender':
                    gender_predictions = models['gender'].predict(img_content, verbose=0)[0, :]
                    obj['gender'] = {}
                    for (i, gender_label) in enumerate(Gender.labels):
                        gender_prediction = 100 * gender_predictions[i]
                        obj['gender'][gender_label] = gender_prediction
                    obj['dominant_gender'] = Gender.labels[np.argmax(gender_predictions)]
                elif action == 'race':
                    race_predictions = models['race'].predict(img_content, verbose=0)[0, :]
                    sum_of_predictions = race_predictions.sum()
                    obj['race'] = {}
                    for (i, race_label) in enumerate(Race.labels):
                        race_prediction = 100 * race_predictions[i] / sum_of_predictions
                        obj['race'][race_label] = race_prediction
                    obj['dominant_race'] = Race.labels[np.argmax(race_predictions)]
                obj['region'] = img_region
                obj['face_confidence'] = img_confidence
            resp_objects.append(obj)
    return resp_objects

def find(img_path, db_path, model_name='VGG-Face', distance_metric='cosine', enforce_detection=True, detector_backend='opencv', align=True, normalization='base', silent=False):
    if False:
        return 10
    "\n    This function applies verification several times and find the identities in a database\n\n    Parameters:\n            img_path: exact image path, numpy array (BGR) or based64 encoded image.\n            Source image can have many faces. Then, result will be the size of number of\n            faces in the source image.\n\n            db_path (string): You should store some image files in a folder and pass the\n            exact folder path to this. A database image can also have many faces.\n            Then, all detected faces in db side will be considered in the decision.\n\n            model_name (string): VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID,\n            Dlib, ArcFace, SFace or Ensemble\n\n            distance_metric (string): cosine, euclidean, euclidean_l2\n\n            enforce_detection (bool): The function throws exception if a face could not be detected.\n            Set this to False if you don't want to get exception. This might be convenient for low\n            resolution images.\n\n            detector_backend (string): set face detector backend to opencv, retinaface, mtcnn, ssd,\n            dlib, mediapipe or yolov8.\n\n            align (boolean): alignment according to the eye positions.\n\n            normalization (string): normalize the input image before feeding to model\n\n            silent (boolean): disable some logging and progress bars\n\n    Returns:\n            This function returns list of pandas data frame. Each item of the list corresponding to\n            an identity in the img_path.\n    "
    tic = time.time()
    if os.path.isdir(db_path) is not True:
        raise ValueError('Passed db_path does not exist!')
    target_size = functions.find_target_size(model_name=model_name)
    file_name = f'representations_{model_name}.pkl'
    file_name = file_name.replace('-', '_').lower()
    if path.exists(db_path + '/' + file_name):
        if not silent:
            print(f'WARNING: Representations for images in {db_path} folder were previously stored' + f' in {file_name}. If you added new instances after the creation, then please ' + 'delete this file and call find function again. It will create it again.')
        with open(f'{db_path}/{file_name}', 'rb') as f:
            representations = pickle.load(f)
        if not silent:
            print('There are ', len(representations), ' representations found in ', file_name)
    else:
        employees = []
        for (r, _, f) in os.walk(db_path):
            for file in f:
                if '.jpg' in file.lower() or '.jpeg' in file.lower() or '.png' in file.lower():
                    exact_path = r + '/' + file
                    employees.append(exact_path)
        if len(employees) == 0:
            raise ValueError('There is no image in ', db_path, ' folder! Validate .jpg or .png files exist in this path.')
        representations = []
        pbar = tqdm(range(0, len(employees)), desc='Finding representations', disable=silent)
        for index in pbar:
            employee = employees[index]
            img_objs = functions.extract_faces(img=employee, target_size=target_size, detector_backend=detector_backend, grayscale=False, enforce_detection=enforce_detection, align=align)
            for (img_content, _, _) in img_objs:
                embedding_obj = represent(img_path=img_content, model_name=model_name, enforce_detection=enforce_detection, detector_backend='skip', align=align, normalization=normalization)
                img_representation = embedding_obj[0]['embedding']
                instance = []
                instance.append(employee)
                instance.append(img_representation)
                representations.append(instance)
        with open(f'{db_path}/{file_name}', 'wb') as f:
            pickle.dump(representations, f)
        if not silent:
            print(f'Representations stored in {db_path}/{file_name} file.' + 'Please delete this file when you add new identities in your database.')
    df = pd.DataFrame(representations, columns=['identity', f'{model_name}_representation'])
    target_objs = functions.extract_faces(img=img_path, target_size=target_size, detector_backend=detector_backend, grayscale=False, enforce_detection=enforce_detection, align=align)
    resp_obj = []
    for (target_img, target_region, _) in target_objs:
        target_embedding_obj = represent(img_path=target_img, model_name=model_name, enforce_detection=enforce_detection, detector_backend='skip', align=align, normalization=normalization)
        target_representation = target_embedding_obj[0]['embedding']
        result_df = df.copy()
        result_df['source_x'] = target_region['x']
        result_df['source_y'] = target_region['y']
        result_df['source_w'] = target_region['w']
        result_df['source_h'] = target_region['h']
        distances = []
        for (index, instance) in df.iterrows():
            source_representation = instance[f'{model_name}_representation']
            if distance_metric == 'cosine':
                distance = dst.findCosineDistance(source_representation, target_representation)
            elif distance_metric == 'euclidean':
                distance = dst.findEuclideanDistance(source_representation, target_representation)
            elif distance_metric == 'euclidean_l2':
                distance = dst.findEuclideanDistance(dst.l2_normalize(source_representation), dst.l2_normalize(target_representation))
            else:
                raise ValueError(f'invalid distance metric passes - {distance_metric}')
            distances.append(distance)
        result_df[f'{model_name}_{distance_metric}'] = distances
        threshold = dst.findThreshold(model_name, distance_metric)
        result_df = result_df.drop(columns=[f'{model_name}_representation'])
        result_df = result_df[result_df[f'{model_name}_{distance_metric}'] <= threshold]
        result_df = result_df.sort_values(by=[f'{model_name}_{distance_metric}'], ascending=True).reset_index(drop=True)
        resp_obj.append(result_df)
    toc = time.time()
    if not silent:
        print('find function lasts ', toc - tic, ' seconds')
    return resp_obj

def represent(img_path, model_name='VGG-Face', enforce_detection=True, detector_backend='opencv', align=True, normalization='base'):
    if False:
        return 10
    '\n    This function represents facial images as vectors. The function uses convolutional neural\n    networks models to generate vector embeddings.\n\n    Parameters:\n            img_path (string): exact image path. Alternatively, numpy array (BGR) or based64\n            encoded images could be passed. Source image can have many faces. Then, result will\n            be the size of number of faces appearing in the source image.\n\n            model_name (string): VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib,\n            ArcFace, SFace\n\n            enforce_detection (boolean): If no face could not be detected in an image, then this\n            function will return exception by default. Set this to False not to have this exception.\n            This might be convenient for low resolution images.\n\n            detector_backend (string): set face detector backend to opencv, retinaface, mtcnn, ssd,\n            dlib, mediapipe or yolov8.\n\n            align (boolean): alignment according to the eye positions.\n\n            normalization (string): normalize the input image before feeding to model\n\n    Returns:\n            Represent function returns a list of object with multidimensional vector (embedding).\n            The number of dimensions is changing based on the reference model.\n            E.g. FaceNet returns 128 dimensional vector; VGG-Face returns 2622 dimensional vector.\n    '
    resp_objs = []
    model = build_model(model_name)
    target_size = functions.find_target_size(model_name=model_name)
    if detector_backend != 'skip':
        img_objs = functions.extract_faces(img=img_path, target_size=target_size, detector_backend=detector_backend, grayscale=False, enforce_detection=enforce_detection, align=align)
    else:
        if isinstance(img_path, str):
            img = functions.load_image(img_path)
        elif type(img_path).__module__ == np.__name__:
            img = img_path.copy()
        else:
            raise ValueError(f'unexpected type for img_path - {type(img_path)}')
        if len(img.shape) == 4:
            img = img[0]
        if len(img.shape) == 3:
            img = cv2.resize(img, target_size)
            img = np.expand_dims(img, axis=0)
        img_region = [0, 0, img.shape[1], img.shape[0]]
        img_objs = [(img, img_region, 0)]
    for (img, region, confidence) in img_objs:
        img = functions.normalize_input(img=img, normalization=normalization)
        if 'keras' in str(type(model)):
            embedding = model(img, training=False).numpy()[0].tolist()
        else:
            embedding = model.predict(img)[0].tolist()
        resp_obj = {}
        resp_obj['embedding'] = embedding
        resp_obj['facial_area'] = region
        resp_obj['face_confidence'] = confidence
        resp_objs.append(resp_obj)
    return resp_objs

def stream(db_path='', model_name='VGG-Face', detector_backend='opencv', distance_metric='cosine', enable_face_analysis=True, source=0, time_threshold=5, frame_threshold=5):
    if False:
        i = 10
        return i + 15
    '\n    This function applies real time face recognition and facial attribute analysis\n\n    Parameters:\n            db_path (string): facial database path. You should store some .jpg files in this folder.\n\n            model_name (string): VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib,\n            ArcFace, SFace\n\n            detector_backend (string): opencv, retinaface, mtcnn, ssd, dlib, mediapipe or yolov8.\n\n            distance_metric (string): cosine, euclidean, euclidean_l2\n\n            enable_facial_analysis (boolean): Set this to False to just run face recognition\n\n            source: Set this to 0 for access web cam. Otherwise, pass exact video path.\n\n            time_threshold (int): how many second analyzed image will be displayed\n\n            frame_threshold (int): how many frames required to focus on face\n\n    '
    if time_threshold < 1:
        raise ValueError('time_threshold must be greater than the value 1 but you passed ' + str(time_threshold))
    if frame_threshold < 1:
        raise ValueError('frame_threshold must be greater than the value 1 but you passed ' + str(frame_threshold))
    realtime.analysis(db_path, model_name, detector_backend, distance_metric, enable_face_analysis, source=source, time_threshold=time_threshold, frame_threshold=frame_threshold)

def extract_faces(img_path, target_size=(224, 224), detector_backend='opencv', enforce_detection=True, align=True, grayscale=False):
    if False:
        i = 10
        return i + 15
    '\n    This function applies pre-processing stages of a face recognition pipeline\n    including detection and alignment\n\n    Parameters:\n            img_path: exact image path, numpy array (BGR) or base64 encoded image.\n            Source image can have many face. Then, result will be the size of number\n            of faces appearing in that source image.\n\n            target_size (tuple): final shape of facial image. black pixels will be\n            added to resize the image.\n\n            detector_backend (string): face detection backends are retinaface, mtcnn,\n            opencv, ssd or dlib\n\n            enforce_detection (boolean): function throws exception if face cannot be\n            detected in the fed image. Set this to False if you do not want to get\n            an exception and run the function anyway.\n\n            align (boolean): alignment according to the eye positions.\n\n            grayscale (boolean): extracting faces in rgb or gray scale\n\n    Returns:\n            list of dictionaries. Each dictionary will have facial image itself,\n            extracted area from the original image and confidence score.\n\n    '
    resp_objs = []
    img_objs = functions.extract_faces(img=img_path, target_size=target_size, detector_backend=detector_backend, grayscale=grayscale, enforce_detection=enforce_detection, align=align)
    for (img, region, confidence) in img_objs:
        resp_obj = {}
        if len(img.shape) == 4:
            img = img[0]
        resp_obj['face'] = img[:, :, ::-1]
        resp_obj['facial_area'] = region
        resp_obj['confidence'] = confidence
        resp_objs.append(resp_obj)
    return resp_objs

@deprecated(version='0.0.78', reason='Use DeepFace.extract_faces instead of DeepFace.detectFace')
def detectFace(img_path, target_size=(224, 224), detector_backend='opencv', enforce_detection=True, align=True):
    if False:
        while True:
            i = 10
    '\n    Deprecated function. Use extract_faces for same functionality.\n\n    This function applies pre-processing stages of a face recognition pipeline\n    including detection and alignment\n\n    Parameters:\n            img_path: exact image path, numpy array (BGR) or base64 encoded image.\n            Source image can have many face. Then, result will be the size of number\n            of faces appearing in that source image.\n\n            target_size (tuple): final shape of facial image. black pixels will be\n            added to resize the image.\n\n            detector_backend (string): face detection backends are retinaface, mtcnn,\n            opencv, ssd or dlib\n\n            enforce_detection (boolean): function throws exception if face cannot be\n            detected in the fed image. Set this to False if you do not want to get\n            an exception and run the function anyway.\n\n            align (boolean): alignment according to the eye positions.\n\n            grayscale (boolean): extracting faces in rgb or gray scale\n\n    Returns:\n            detected and aligned face as numpy array\n\n    '
    print('⚠️ Function detectFace is deprecated. Use extract_faces instead.')
    face_objs = extract_faces(img_path=img_path, target_size=target_size, detector_backend=detector_backend, enforce_detection=enforce_detection, align=align, grayscale=False)
    extracted_face = None
    if len(face_objs) > 0:
        extracted_face = face_objs[0]['face']
    return extracted_face
functions.initialize_folder()

def cli():
    if False:
        for i in range(10):
            print('nop')
    '\n    command line interface function will be offered in this block\n    '
    import fire
    fire.Fire()