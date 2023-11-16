import prediction
import nlp_rev2
import client
import extraction
import file_receive
import os
import record_vid_mode2
import shutil
import time


def transfer_file_by_name(source_directory, destination_directory, prefix):
    for filename in os.listdir(source_directory):
        if filename.startswith(prefix):
            source_path = os.path.join(source_directory, filename)
            destination_path = os.path.join(destination_directory, filename)
            shutil.move(source_path, destination_path)


def transfer_files_with_prefix(source_dir, dest_dir, prefix):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    files_to_transfer = [filename for filename in os.listdir(source_dir) if filename.startswith(prefix)]

    for filename in files_to_transfer:
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(dest_dir, filename)
        shutil.move(source_path, dest_path)
        print(f"Transferred: {filename}")


def clean_directory(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        try:
            if os.path.isfile(filepath):
                os.remove(filepath)
        except Exception as e:
            print(f"Error deleting file: {filepath}")


def num_files(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    num_dir = len(files)
    return num_dir


def clear_file(filename):
    with open(filename, "w") as f:
        pass  # Do nothing


def delete_last_file(directory):

    # Get a list of all files in the directory
    file_list = os.listdir(directory)

    # Find the name of the last file
    last_file_name = sorted(file_list)[-1]

    # Delete the last file
    os.remove(os.path.join(directory, last_file_name))


def rearrange_lines(input_filename, word_count_list, output_filename):
    with open(input_filename, 'r') as f:
        lines = f.read().splitlines()

    output_lines = []
    index = 0

    for count in word_count_list:
        if index + count <= len(lines):
            output_lines.append(' '.join(lines[index:index + count]))
            index += count
        else:
            print(f"Warning: Not enough remaining words for line with {count} words.")

    with open(output_filename, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')


def combine_words(input_file_path, output_file_path):
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    words = [line.strip() for line in lines]

    combined_sentence = ' '.join(words)

    with open(output_file_path, 'w') as file:
        file.write(combined_sentence)


def wait_for_receive(directory_path):
    while True:
        if not os.listdir(directory_path):
            print("Directory is empty. Waiting for files...")
            time.sleep(1)  # Sleep for 1 second before checking again
        else:
            print("Files detected in the directory. Waiting a few seconds and breaking out.")
            time.sleep(7)  # Wait 1 second
            break



def main():

    dir_vid = "/home/tester/finalProject/videos"
    translated_txt = "/home/tester/finalProject/translated_content.txt"
    dir_pkl = "/home/tester/finalProject/videos_after"
    switch_mode_flag = "/home/tester/finalProject/switch_mode_flag.txt"
    mode2_vid = "/home/tester/finalProject/mode2_main_video/output.mp4"
    dir_vid_mode2 = "/home/tester/finalProject/mode2_videos/"
    dir_vid_desktop = "/home/tester/Desktop/videos/"
    dir_pending = "/home/tester/finalProject/pending_videos"
    mode1_txt = "/home/tester/finalProject/mode1_index.txt"
    dir_pkl_desktop = "/home/tester/Desktop/videos_pkl"

    with open(mode1_txt, 'r') as file:
        content = file.read()
        mode1_index = int(content)

    with open(switch_mode_flag, 'r') as file:
        mode_flag = file.read()

    clear_file(translated_txt)
    prefix = f"out{mode1_index}"

    if mode_flag == "1":
        print("MODE 2 IS UP")
        file_receive.main() # save main video in mode2_main_video
        word_count_list = record_vid_mode2.record_delete_from_video(mode2_vid, dir_vid_mode2)# takes main video from mode2_main_video and save individual videos in mode2_videos
        extraction.main(dir_vid_mode2) #takes videos from mode2_videos and stores pkl files in videos_after
        clean_directory(dir_vid_mode2) #clean mode2_videos
        prediction.translation() #takes pkl from videos_after and translates
        clean_directory(dir_pkl) #clean videos_after
        rearrange_lines(translated_txt, word_count_list, translated_txt) # process of client and NLP
        translated_sentence = nlp_rev2.main(translated_txt, translated_txt)
        client.main(translated_sentence)
    else:
        print("MODE 1 IS UP")
        wait_for_receive(dir_pending)
        time.sleep(1)
        with open(mode1_txt, "w") as file:
            file.write(f"{str(mode1_index + 1)}")
        transfer_files_with_prefix(dir_pending, dir_vid, prefix)
        extraction.main(dir_vid)
        clean_directory(dir_vid)
        prediction.translation()
        clean_directory(dir_pkl)
        combine_words(translated_txt, translated_txt)
        translated_sentence = nlp_rev2.main(translated_txt, translated_txt)
        client.main(translated_sentence)


if __name__ == "__main__":
    main()

