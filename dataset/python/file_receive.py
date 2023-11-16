import socket
import os
import record_vid_mode1

counter = 1


def receive_file(client, file_path, file_size):
    client.send(b"ACK")  # Send acknowledgment back to client to start receiving the file

    # Receive the actual file data from the client
    with open(file_path, "wb") as file:
        received_bytes = 0
        while received_bytes < file_size:
            data = client.recv(1024)
            received_bytes += len(data)
            file.write(data)

    print(f"File received: {file_path}")
    client.send(b"ACK")  # Send acknowledgment back to client for successful file reception


def fileserver(destination_directory):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("192.168.1.131", 9999))
    print("Server is listening on 192.168.1.131:9999")
    server.listen()

    client, addr = server.accept()
    print(f"Connection established with {addr}")

    while True:
        data = client.recv(1024).decode()
        if data == "<END>":
            break

        file_name, file_size = data.split(",")
        file_size = int(file_size)

        file_path = os.path.join(destination_directory, file_name)

        receive_file(client, file_path, file_size)

    client.close()
    server.close()


def main():
    global counter
    switch_mode_flag = "/home/tester/finalProject/switch_mode_flag.txt"

    with open(switch_mode_flag, 'r') as file:
        mode_flag = file.read()

    if mode_flag == "1":
        print("MODE 2 FILE RECEIVE")
        destination_dir_mode2 = "/home/tester/finalProject/mode2_main_video"
        fileserver(destination_dir_mode2)
    else:
        out_prefix = f"out{counter}"
        input_video_path = "/home/tester/finalProject/mode1_main_video/output.mp4"
        pending_dir = "/home/tester/finalProject/pending_videos/"
        output_dir = "/home/tester/Desktop/videos"
        print("MODE 1 FILE RECEIVE")
        destination_dir_mode1 = "/home/tester/finalProject/mode1_main_video"
        fileserver(destination_dir_mode1)
        record_vid_mode1.record_delete_from_video(input_video_path, pending_dir, out_prefix)
        counter = counter + 1


if __name__ == "__main__":
    while True:
        main()