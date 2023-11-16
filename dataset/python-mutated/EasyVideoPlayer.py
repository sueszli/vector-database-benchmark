import cv2
import os
from pathlib import Path
from ffpyplayer.player import MediaPlayer
video_name = input('Name of the video file that you want to play:    ')
video_directory_guess = input('Directory that may contain the video:    ')

def find_the_video(file_name, directory_name):
    if False:
        return 10
    files_found = []
    for (path, subdirs, files) in os.walk(directory_name):
        for name in files:
            if file_name == name:
                file_path = os.path.join(path, name)
                files_found.append(file_path)
    print(files_found)
    return files_found[0]
video_directory = Path(find_the_video(video_name, video_directory_guess))
new_working_directory = video_directory.parent
os.chdir(new_working_directory)
video_path = find_the_video(video_name, video_directory_guess)

def PlayVideo(video_path):
    if False:
        return 10
    video = cv2.VideoCapture(video_path)
    player = MediaPlayer(video_path)
    while True:
        (grabbed, frame) = video.read()
        (audio_frame, val) = player.get_frame()
        if not grabbed:
            print('End of video')
            break
        if cv2.waitKey(28) & 255 == ord('q'):
            break
        cv2.imshow('Video', frame)
        if val != 'eof' and audio_frame is not None:
            (img, t) = audio_frame
    video.release()
    cv2.destroyAllWindows()
PlayVideo(video_path)