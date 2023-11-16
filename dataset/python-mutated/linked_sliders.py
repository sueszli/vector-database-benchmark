import streamlit as st
to_celsius = lambda fahrenheit: (fahrenheit - 32) * 5.0 / 9.0
to_fahrenheit = lambda celsius: 9.0 / 5.0 * celsius + 32
(MIN_CELSIUS, MAX_CELSIUS) = (-100.0, 100.0)
state = st.session_state
if 'celsius' not in st.session_state:
    state.celsius = MIN_CELSIUS
    state.fahrenheit = to_fahrenheit(MIN_CELSIUS)

def celsius_changed():
    if False:
        while True:
            i = 10
    state.fahrenheit = to_fahrenheit(state.celsius)

def fahrenheit_changed():
    if False:
        for i in range(10):
            print('nop')
    state.celsius = to_celsius(state.fahrenheit)
st.slider('Celsius', min_value=MIN_CELSIUS, max_value=MAX_CELSIUS, on_change=celsius_changed, key='celsius')
st.slider('Fahrenheit', min_value=to_fahrenheit(MIN_CELSIUS), max_value=to_fahrenheit(MAX_CELSIUS), on_change=fahrenheit_changed, key='fahrenheit')
st.write('Celsius', state.celsius)
st.write('Fahrenheit', state.fahrenheit)