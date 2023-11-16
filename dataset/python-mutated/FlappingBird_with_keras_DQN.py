from __future__ import print_function
import argparse
from collections import deque
import json
import numpy as np
import os
import random
import requests
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import sys
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve
import game.wrapped_flappy_bird as game
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam
GAME = 'bird'
CONFIG = 'nothreshold'
ACTIONS = 2
GAMMA = 0.99
OBSERVATION = 320.0
EXPLORE = 3000000.0
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.1
REPLAY_MEMORY = 50000
BATCH = 32
FRAME_PER_ACTION = 1
LEARNING_RATE = 0.0001
NUMRUNS = 400
PRETRAINED_MODEL_URL_DEFAULT = 'https://cntk.ai/Models/FlappingBird_keras/FlappingBird_model.h5.model'
PRETRAINED_MODEL_FNAME = 'FlappingBird_model.h5'
(img_rows, img_cols) = (80, 80)
img_channels = 4

def pretrained_model_download(url, filename, max_retries=3):
    if False:
        print('Hello World!')
    'Download the file unless it already exists, with retry. Throws if all retries fail.'
    if os.path.exists(filename):
        print('Reusing locally cached: ', filename)
    else:
        print('Starting download of {} to {}'.format(url, filename))
        retry_cnt = 0
        while True:
            try:
                urlretrieve(url, filename)
                print('Download completed.')
                return
            except:
                retry_cnt += 1
                if retry_cnt == max_retries:
                    raise Exception('Exceeded maximum retry count, aborting.')
                print('Failed to download, retrying.')
                time.sleep(np.random.randint(1, 10))

def buildmodel():
    if False:
        for i in range(10):
            print('nop')
    print('Now we build the model')
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same', input_shape=(img_rows, img_cols, img_channels)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(2))
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse', optimizer=adam)
    print('We finish building the model')
    return model

def trainNetwork(model, args, pretrained_model_url=None, internal_testing=False):
    if False:
        i = 10
        return i + 15
    print(args)
    if not pretrained_model_url:
        pretrained_model_url = PRETRAINED_MODEL_URL_DEFAULT
    else:
        pretrained_model_url = pretrained_model_url
    pretrained_model_fname = PRETRAINED_MODEL_FNAME
    game_state = game.GameState()
    D = deque()
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    (x_t, r_0, terminal) = game_state.frame_step(do_nothing)
    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t, (80, 80))
    x_t = skimage.exposure.rescale_intensity(x_t, out_range=(0, 255))
    if internal_testing:
        x_t = np.random.rand(x_t.shape[0], x_t.shape[1])
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2).astype(np.float32)
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
    if args['mode'] == 'Run':
        OBSERVE = 999999999
        epsilon = FINAL_EPSILON
        print('Now we load weight')
        pretrained_model_download(pretrained_model_url, pretrained_model_fname)
        model.load_weights(pretrained_model_fname)
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        print('Weight load successfully')
    else:
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON
    t = 0
    while True:
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([ACTIONS])
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print('----------Random Action----------')
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                q = model.predict(s_t)
                max_Q = np.argmax(q)
                action_index = max_Q
                a_t[max_Q] = 1
        if args['mode'] == 'Run' and t > NUMRUNS:
            break
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        (x_t1_colored, r_t, terminal) = game_state.frame_step(a_t)
        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1, (80, 80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1).astype(np.float32)
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()
        if t > OBSERVE:
            minibatch = random.sample(D, BATCH)
            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3])).astype(np.float32)
            print(inputs.shape)
            targets = np.zeros((inputs.shape[0], ACTIONS)).astype(np.float32)
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]
                inputs[i:i + 1] = state_t
                targets[i] = model.predict(state_t)
                Q_sa = model.predict(state_t1)
                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)
            loss += model.train_on_batch(inputs, targets)
        s_t = s_t1
        t = t + 1
        if t % 1000 == 0:
            print('Now we save model')
            model.save_weights(pretrained_model_fname, overwrite=True)
            with open('model.json', 'w') as outfile:
                json.dump(model.to_json(), outfile)
        state = ''
        if t <= OBSERVE:
            if internal_testing:
                return 0
            else:
                state = 'observe'
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = 'explore'
        else:
            state = 'train'
        print('TIMESTEP', t, '/ STATE', state, '/ EPSILON', epsilon, '/ ACTION', action_index, '/ REWARD', r_t, '/ Q_MAX ', np.max(Q_sa), '/ Loss ', loss)
    print('Episode finished!')
    print('************************')

def playGame(args):
    if False:
        while True:
            i = 10
    model = buildmodel()
    trainNetwork(model, args)

def main():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m', '--mode', help='Train / Run', required=True)
    args = vars(parser.parse_args())
    playGame(args)
if __name__ == '__main__':
    main()