import requests
import streamlit as st

@st.experimental_memo
def audio():
    if False:
        print('Hello World!')
    url = 'https://www.w3schools.com/html/horse.ogg'
    file = requests.get(url).content
    st.audio(file)

@st.experimental_memo
def video():
    if False:
        return 10
    url = 'https://www.w3schools.com/html/mov_bbb.mp4'
    file = requests.get(url).content
    st.video(file)
audio()
video()