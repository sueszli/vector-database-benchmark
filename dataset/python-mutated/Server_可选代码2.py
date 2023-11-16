from socket import *
from _thread import *
import threading
print_lock = threading.Lock()

def threaded(c):
    if False:
        while True:
            i = 10
    try:
        message = c.recv(1024)
        filename = message.split()[1]
        f = open(filename[1:])
        outputdata = f.read()
        header = 'HTTP/1.1 200 OK \nConnection: close\n' + 'Content0Length: {}\n'.format(len(outputdata)) + 'Content-Type: text/html\n\n'
        c.send(header.encode())
        for i in range(0, len(outputdata)):
            c.send(outputdata[i].encode())
        c.close()
    except IOError:
        header = 'HTTP/1.1 404 Not Found'
        c.send(header.encode())
        c.close()

def main():
    if False:
        while True:
            i = 10
    serverSocket = socket(AF_INET, SOCK_STREAM)
    serverSocket.bind(('', 81))
    serverSocket.listen(1)
    while True:
        try:
            print('Ready to server...')
            (connectionSocket, addr) = serverSocket.accept()
            start_new_thread(threaded, (connectionSocket,))
        except:
            print('Exit')
            break
    serverSocket.close()
if __name__ == '__main__':
    main()