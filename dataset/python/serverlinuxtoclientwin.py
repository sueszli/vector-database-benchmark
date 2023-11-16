import socket

# SERVER

dataSend = "Hello"

print("Server Application Started")
print(f"Attempting to send: {dataSend}")

server = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
server.bind(("A8:7E:EA:EC:52:51", 4)) # MAC ADDRESS OF LINUX
server.listen(1)

client, addr = server.accept()

try:
    while True:
        message = dataSend
        client.send(message.encode('utf-8'))

except OSError as e:
    pass

client.close()
server.close()
