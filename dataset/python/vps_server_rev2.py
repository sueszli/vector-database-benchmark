import socket
import threading

HOST = "192.168.1.131"
PORT = 1234

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))

server.listen()

clients = []
nicknames = []


def mode2():
    switch_mode_2_flag = "/home/tester/finalProject/switch_mode_flag.txt"
    with open(switch_mode_2_flag, 'w') as file:
        file.write("1")


def mode1():
    switch_mode_2_flag = "/home/tester/finalProject/switch_mode_flag.txt"
    with open(switch_mode_2_flag, 'w') as file:
        file.write("0")


def broadcast(message):
    for client in clients:
        client.send(message)


def handle_connection(client, nickname):
    stop = False
    while not stop:
        try:
            message = client.recv(1024).decode('utf-8')
            if message == f"{nickname}: CHANGE_MODE_2":
                print("MODE 2 detected")
                mode2()
                message_mode2 = "Mode 2 Enabled successfully, Upload video to server using raspberry!"
                broadcast(message_mode2.encode('utf-8'))
            elif message == f"{nickname}: CHANGE_MODE_1":
                print("MODE 1 detected")
                mode1()
                message_mode1 = "Mode 1 Enabled successfully - Happy translating!"
                broadcast(message_mode1.encode('utf-8'))
            else:
                broadcast(message.encode('utf-8'))
        except:
            index = clients.index(client)
            clients.remove(client)
            nickname = nicknames[index]
            nicknames.remove(nickname)
            broadcast(f"{nickname} left the chat! \n".encode('utf-8'))
            stop = True


def main():
    print("192.168.1.131:1234 - Chat Room Host Server")
    while True:
        client, addr = server.accept()
        print(f"Connected to {addr}")

        client.send("NICK".encode('utf-8'))

        nickname = client.recv(1024).decode('utf-8')
        nicknames.append(nickname)
        clients.append(client)
        print(f"Nickname is {nickname}")

        broadcast(f"{nickname} joined the chat!".encode('utf-8'))

        client.send("You are now connected! \n".encode('utf-8'))

        thread = threading.Thread(target=handle_connection, args=(client, nickname))
        thread.start()

        welcome_message = "Welcome to the chat room! - Please type: CHANGE_MODE_2 or CHANGE_MODE_1 to change modes"
        client.send(welcome_message.encode('utf-8'))


if __name__ == "__main__":
    main()
