import PySimpleGUI as sg
from chatterbot import ChatBot
import chatterbot.utils
from gtts import gTTS
from pygame import mixer
import time
import os
'\nDemo_Chatterbot.py\n\n\nNote - this code was written using version 0.8.7 of Chatterbot... to install:\n\npython -m pip install chatterbot==0.8.7\n\nIt still runs fine with the old version. \n\nA GUI wrapped arouind the Chatterbot package.\nThe GUI is used to show progress bars during the training process and\nto collect user input that is sent to the chatbot.  The reply is displayed in the GUI window\n'
sg.theme('NeutralBlue')
MAX_PROG_BARS = 20
bars = []
texts = []
training_layout = [[sg.Text('TRAINING PROGRESS', size=(20, 1), font=('Helvetica', 17))]]
for i in range(MAX_PROG_BARS):
    bars.append(sg.ProgressBar(100, size=(30, 4)))
    texts.append(sg.Text(' ' * 20, size=(20, 1), justification='right'))
    training_layout += [[texts[i], bars[i]]]
training_window = sg.Window('Training', training_layout)
current_bar = 0

def print_progress_bar(description, iteration_counter, total_items, progress_bar_length=20):
    if False:
        return 10
    global current_bar
    global bars
    global texts
    global training_window
    (button, values) = training_window.read(timeout=0)
    if button is None:
        return
    if bars[current_bar].update_bar(iteration_counter, max=total_items) is False:
        return
    texts[current_bar].update(description)
    if iteration_counter == total_items:
        current_bar += 1

def speak(text):
    if False:
        while True:
            i = 10
    global i
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save('speech{}.mp3'.format(i % 2))
    mixer.music.load('speech{}.mp3'.format(i % 2))
    mixer.music.play()
    while mixer.music.get_busy():
        time.sleep(0.1)
    mixer.stop()
    i += 1
i = 0
mixer.init()
chatterbot.utils.print_progress_bar = print_progress_bar
chatbot = ChatBot('Ron Obvious', trainer='chatterbot.trainers.ChatterBotCorpusTrainer')
chatbot.train('chatterbot.corpus.english')
layout = [[sg.Multiline(size=(80, 20), reroute_stdout=True, echo_stdout_stderr=True)], [sg.MLine(size=(70, 5), key='-MLINE IN-', enter_submits=True, do_not_clear=False), sg.Button('SEND', bind_return_key=True), sg.Button('EXIT')]]
window = sg.Window('Chat Window', layout, default_element_size=(30, 2))
while True:
    (event, values) = window.read()
    if event != 'SEND':
        break
    string = values['-MLINE IN-'].rstrip()
    print('  ' + string)
    response = chatbot.get_response(values['-MLINE IN-'].rstrip())
    print(response)
window.close()