"""
Retos Semanales ‘23
Reto #25: EL CÓDIGO KONAMI
MEDIA | Publicación: 19/06/23 | Resolución: 26/06/23

 * Crea un programa que detecte cuando el famoso "Código Konami" se ha introducido
 * correctamente desde el teclado. 
 * Si sucede esto, debe notificarse mostrando un mensaje en la terminal.
"""
import typer
from rich import print
import random
from pynput.keyboard import Key, KeyCode, Listener
PHRASES = ['Ok... interesante... cuéntame más.', 'Muy bien, suena cool... ¿qué más?', 'Bien, bien... qué curioso... sigue contandome :)', 'Cool!, te sigo leyendo...', 'Me gusta leerte :) ... continua por favor...']
KONAMI_CODE = (Key.up, Key.up, Key.down, Key.down, Key.left, Key.right, Key.left, Key.right, KeyCode.from_char('b'), KeyCode.from_char('a'))
KONAMI_CODE_LETTERING = f'[bold yellow]\n██╗  ██╗ ██████╗ ███╗   ██╗ █████╗ ███╗   ███╗██╗     ██████╗ ██████╗ ██████╗ ███████╗\n██║ ██╔╝██╔═══██╗████╗  ██║██╔══██╗████╗ ████║██║    ██╔════╝██╔═══██╗██╔══██╗██╔════╝\n█████╔╝ ██║   ██║██╔██╗ ██║███████║██╔████╔██║██║    ██║     ██║   ██║██║  ██║█████╗  \n██╔═██╗ ██║   ██║██║╚██╗██║██╔══██║██║╚██╔╝██║██║    ██║     ██║   ██║██║  ██║██╔══╝  \n██║  ██╗╚██████╔╝██║ ╚████║██║  ██║██║ ╚═╝ ██║██║    ╚██████╗╚██████╔╝██████╔╝███████╗\n╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝     ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝\n'
welcomePending = True
consecutiveCodeKeysPressed = 0
_exit = False

def main_menu():
    if False:
        for i in range(10):
            print('nop')
    global welcomePending
    if welcomePending:
        print('[green]\nBienvenido al Script de [yellow]El Código Konami[/yellow], esto es algo similar a un diario, un espacion donde me puedes contar acerca de ti, de tu día, y de todo lo que quieras, o presiona la tecla ESCAPE para salir.[/green] 😀')
        welcomePending = False
    else:
        print(f'[dodger_blue2]\n{random.choice(PHRASES)}')

def key_listener(key):
    if False:
        print('Hello World!')
    global _exit
    global consecutiveCodeKeysPressed
    if key == KONAMI_CODE[consecutiveCodeKeysPressed]:
        consecutiveCodeKeysPressed += 1
    elif consecutiveCodeKeysPressed > 0:
        consecutiveCodeKeysPressed = 0
    if consecutiveCodeKeysPressed == len(KONAMI_CODE):
        print(f'\n\n[yellow]Ha ingresado el Código Konami!, se ha desbloqueado una funcionalidad secreta :)')
        print(KONAMI_CODE_LETTERING)
        consecutiveCodeKeysPressed = 0
    if key == Key.esc:
        print(f'\n\n[yellow]Has presionado la tecla ESCAPE, el programa terminará cuando presiones ENTER.\n')
        _exit = True
        return False

def main():
    if False:
        while True:
            i = 10
    global _exit
    print('[bold green]\n*** Reto #25: EL CÓDIGO KONAMI - By @ClarkCodes ***')
    listener = Listener(on_release=key_listener)
    listener.start()
    while True:
        main_menu()
        print('\n[bold green]Respuesta: ', end='')
        try:
            user_answer = input('')
            if _exit:
                print('[green]\n✅ Esto ha sido todo por hoy.\n❤ Muchas gracias por ejecutar este Script, hasta la próxima...💻 Happy Coding!,👋🏼 bye :D\n😎 Clark.')
                break
        except ValueError as ve:
            print("\n❌ Opción ingresada no disponible, solo se admiten números enteros positivos mayores o iguales a 2, o la letra 'q' si deseas salir, verifique nuevamente.")
            print(ve)
        except Exception as ex:
            print('\n❌ Oops... algo no ha salido bien, revise nuevamente por favor.')
            print(ex)
if __name__ == '__main__':
    typer.run(main)