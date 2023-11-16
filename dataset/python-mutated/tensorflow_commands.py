import time
import speech_recognition as sr
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
r = sr.Recognizer()
m = sr.Microphone()
with m as source:
    r.adjust_for_ambient_noise(source)

def callback(recognizer, audio):
    if False:
        return 10
    try:
        spoken = recognizer.recognize_tensorflow(audio, tensor_graph='speech_recognition/tensorflow-data/conv_actions_frozen.pb', tensor_label='speech_recognition/tensorflow-data/conv_actions_labels.txt')
        print(spoken)
    except sr.UnknownValueError:
        print('Tensorflow could not understand audio')
    except sr.RequestError as e:
        print('Could not request results from Tensorflow service; {0}'.format(e))
stop_listening = r.listen_in_background(m, callback, phrase_time_limit=0.6)
time.sleep(100)