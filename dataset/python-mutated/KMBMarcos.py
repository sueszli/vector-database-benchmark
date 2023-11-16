"""
 Crea un programa que detecte cuando el famoso "Código Konami" se ha introducido
 correctamente desde el teclado. 
 Si sucede esto, debe notificarse mostrando un mensaje en la terminal.
"""
import keyboard as kb
codeKonamiHealth = ['up', 'down', 'a', 'b']
codeKonamiAtk = ['up', 'up', 'a', 'a']
codeKonameDef = ['down', 'down', 'b', 'b']
keyPressed = []

def verifyCode():
    if False:
        print('Hello World!')
    if keyPressed == codeKonameDef:
        print('Defense Up!')
    elif keyPressed == codeKonamiAtk:
        print('Atack Up!')
    elif keyPressed == codeKonamiHealth:
        print('Health Up!')

def on_key_press(event):
    if False:
        print('Hello World!')
    key = event.name
    keyPressed.append(key)
    print(f'The key pressed is: {key}')
    verifyCode()
kb.on_press(on_key_press)
kb.wait('esc')