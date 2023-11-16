import time
import speech_recognition as sr

def callback(recognizer, audio):
    if False:
        while True:
            i = 10
    try:
        print('Google Speech Recognition thinks you said ' + recognizer.recognize_google(audio))
    except sr.UnknownValueError:
        print('Google Speech Recognition could not understand audio')
    except sr.RequestError as e:
        print('Could not request results from Google Speech Recognition service; {0}'.format(e))
r = sr.Recognizer()
m = sr.Microphone()
with m as source:
    r.adjust_for_ambient_noise(source)
stop_listening = r.listen_in_background(m, callback)
for _ in range(50):
    time.sleep(0.1)
stop_listening(wait_for_stop=False)
while True:
    time.sleep(0.1)