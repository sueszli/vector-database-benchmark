from threading import Thread
from queue import Queue
import speech_recognition as sr
r = sr.Recognizer()
audio_queue = Queue()

def recognize_worker():
    if False:
        for i in range(10):
            print('nop')
    while True:
        audio = audio_queue.get()
        if audio is None:
            break
        try:
            print('Google Speech Recognition thinks you said ' + r.recognize_google(audio))
        except sr.UnknownValueError:
            print('Google Speech Recognition could not understand audio')
        except sr.RequestError as e:
            print('Could not request results from Google Speech Recognition service; {0}'.format(e))
        audio_queue.task_done()
recognize_thread = Thread(target=recognize_worker)
recognize_thread.daemon = True
recognize_thread.start()
with sr.Microphone() as source:
    try:
        while True:
            audio_queue.put(r.listen(source))
    except KeyboardInterrupt:
        pass
audio_queue.join()
audio_queue.put(None)
recognize_thread.join()