import record_vid
import client_send_vid
import raspberry_helpers
import time
import simple_record

if __name__ == "__main__":
    output_file_simple_recorder = "/home/tester/videos/output.mp4"
    directory = "/home/ubuntu/videos/"
    simple_record.record_video(output_file_simple_recorder)
    #record_vid.record(directory)
    print("Initiate file receive script on Server - client script will start in 1 seconds")
    time.sleep(1)
    client_send_vid.fileclient(directory)
    raspberry_helpers.clean_directory(directory)


