import socket

#CLIENT

client = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
client.connect(("2c:db:07:44:5a:74", 4)) # MAC ADDRESS OF WINDOWS

try:
    while True:
        data = client.recv(1024)
        print(f"Received Message: {data.decode('utf-8')}")

except OSError as e:
    pass

client.close()
