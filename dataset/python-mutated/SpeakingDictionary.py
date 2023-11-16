import pyttsx3
from PyDictionary import PyDictionary
import speech_recognition as spr
from gtts import gTTS
import os

class Speak:

    def SpeakWord(self, audio):
        if False:
            print('Hello World!')
        pSpeakEngine = pyttsx3.init('sapi5')
        pVoices = pSpeakEngine.getProperty('voices')
        pSpeakEngine.setProperty('voices', pVoices[0].id)
        pSpeakEngine.say(audio)
        pSpeakEngine.runAndWait()
sRecog = spr.Recognizer()
sMic = spr.Microphone()
with sMic as source:
    print("Speak 'Hello' to initiate Speaking Dictionary!")
    print('----------------------------------------------')
    sRecog.adjust_for_ambient_noise(source, duration=0.2)
    rAudio = sRecog.listen(source)
    szHello = sRecog.recognize_google(rAudio, language='en-US')
    szHello = szHello.lower()
if 'hello' in szHello:
    sSpeak = Speak()
    pDict = PyDictionary()
    print('Which word do you want to find? Please speak slowly.')
    sSpeak.SpeakWord('Which word do you want to find Please speak slowly')
    try:
        sRecog2 = spr.Recognizer()
        sMic2 = spr.Microphone()
        with sMic2 as source2:
            sRecog2.adjust_for_ambient_noise(source2, duration=0.2)
            rAudio2 = sRecog2.listen(source2)
            szInput = sRecog2.recognize_google(rAudio2, language='en-US')
            try:
                print('Did you said ' + szInput + '? Please answer with yes or no.')
                sSpeak.SpeakWord('Did you said ' + szInput + 'Please answer with yes or no')
                sRecog2.adjust_for_ambient_noise(source2, duration=0.2)
                rAudioYN = sRecog2.listen(source2)
                szYN = sRecog2.recognize_google(rAudioYN)
                szYN = szYN.lower()
                if 'yes' in szYN:
                    szMeaning = pDict.meaning(szInput)
                    print('The meaning is ', end='')
                    for i in szMeaning:
                        print(szMeaning[i])
                        sSpeak.SpeakWord('The meaning is' + str(szMeaning[i]))
                else:
                    sSpeak.SpeakWord('I am sorry Please try again')
            except spr.UnknownValueError:
                sSpeak.SpeakWord('Unable to understand the input Please try again')
            except spr.RequestError as e:
                sSpeak.SpeakWord('Unable to provide required output')
    except spr.UnknownValueError:
        sSpeak.SpeakWord('Unable to understand the input Please try again')
    except spr.RequestError as e:
        sSpeak.SpeakWord('Unable to provide required output')