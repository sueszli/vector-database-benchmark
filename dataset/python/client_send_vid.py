import socket
import os


def send_file(client, file_path):
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)

    # Send the file name and size to the server
    client.send(f"{file_name},{file_size}".encode())
    response = client.recv(1024).decode()
    if response == "ACK":
        print(f"Sending file: {file_name}")
    else:
        print(f"Error sending file: {file_name}")
        return

    # Send the actual file data to the server in chunks
    with open(file_path, "rb") as file:
        while True:
            data = file.read(1024)
            if not data:
                break
            sent_bytes = 0
            while sent_bytes < len(data):
                sent_bytes += client.send(data[sent_bytes:])

    # Wait for the server to acknowledge that the file has been received
    response = client.recv(1024).decode()
    if response == "ACK":
        print(f"File sent successfully: {file_name}")
    else:
        print(f"Error sending file: {file_name}")

def fileclient(directory):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("192.168.1.131", 9999))
    print("Connected to the server.")

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            send_file(client, file_path)

    # Send an <END> message to indicate that all files have been sent
    client.send(b"<END>")
    print("All files sent.")
    client.close()


if __name__ == "__main__":
    directory_to_send = r"C:\Users\tshrem\OneDrive - Intel Corporation\Desktop\PyCharmProjectsFolder\BluetoothApp\videos"
    fileclient(directory_to_send)


