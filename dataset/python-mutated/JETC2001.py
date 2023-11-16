"""
  Escribe un programa que reciba un texto y transforme lenguaje natural a
  "lenguaje hacker" (conocido realmente como "leet" o "1337"). Este lenguaje
   se caracteriza por sustituir caracteres alfanuméricos.
  - Utiliza esta tabla (https://www.gamehouse.com/blog/leet-speak-cheat-sheet/) 
    con el alfabeto y los números en "leet".
    (Usa la primera opción de cada transformación. Por ejemplo "4" para la "a")
"""
dic = {'a': '4', 'b': 'l3', 'c': '[', 'd': ')', 'e': '3', 'f': '|=', 'g': '&', 'h': 'h', 'i': '1', 'j': ',_|', 'k': '>|', 'l': '1', 'm': '[V]', 'n': '|\\|', 'o': '0', 'p': '|*', 'q': '(_,)', 'r': 'I2', 's': '5', 't': '7', 'u': '(_)', 'v': '|/', 'w': 'VV', 'x': '><', 'y': 'j', 'z': '2', ' ': ' '}
msg = input('Hi dear, write your message: ')
msg_lower = msg.lower()
cmsg = list(msg_lower)
newMsg = []

def traductor():
    if False:
        while True:
            i = 10
    for i in range(len(cmsg)):
        if cmsg[i] in dic:
            newMsg.append(dic[cmsg[i]])
    return newMsg

def listToString(s):
    if False:
        for i in range(10):
            print('nop')
    str1 = ' '
    return str1.join(s)
if __name__ == '__main__':
    print(listToString(traductor()))