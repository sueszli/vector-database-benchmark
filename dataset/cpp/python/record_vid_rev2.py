import cv2
import mediapipe as mp
import os
import time


def delete_last_index_files(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Create a dictionary to store the latest index for each prefix
    latest_indices = {}

    # Find the latest index for each prefix
    for filename in files:
        prefix, index = filename.split('put')[0], int(filename.split('put')[1].split('.')[0])
        latest_indices[prefix] = max(latest_indices.get(prefix, 0), index)

    # Delete files with the latest index per prefix
    for prefix, latest_index in latest_indices.items():
        file_to_delete = f"{prefix}put{latest_index}.mp4"
        file_path = os.path.join(folder_path, file_to_delete)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_to_delete}")


def record(output_dir):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.75) as hands:
        is_recording = False
        recording_count = 1  # Initialize index to 1
        no_hand_count = 0
        recording_delay = 20

        os.makedirs(output_dir, exist_ok=True)

        start_time_left = None
        start_time_right = None
        stop_timer_left = False
        stop_timer_right = False
        corner_position = None
        out_prefix = 'out1'  # Initialize prefix to 'out1'
        put_count = 1  # Initialize prefix index to 1
        index_reset = False
        last_saved_video = None  # To track the last saved video path

        while True:
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                index_finger_tip_landmark = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                height, width, _ = image.shape
                corner_boundary = int(min(height, width) * 0.25)

                if index_finger_tip_landmark.x * width > (width - corner_boundary) and index_finger_tip_landmark.y * height < corner_boundary:
                    if not stop_timer_right:
                        start_time_right = time.time()
                        stop_timer_right = True
                        corner_position = 'top_right'
                        index_reset = False
                    else:
                        elapsed_time = time.time() - start_time_right
                        if elapsed_time >= 1.5:  # Check if finger is in corner for 1 second
                            put_count += 1
                            out_prefix = f'out{put_count}'
                            index_reset = True
                            print(f'Changed prefix to: {out_prefix}')
                            stop_timer_right = False
                else:
                    stop_timer_right = False

                if index_finger_tip_landmark.x * width < corner_boundary and index_finger_tip_landmark.y * height < corner_boundary:
                    if not stop_timer_left:
                        start_time_left = time.time()
                        stop_timer_left = True
                        corner_position = 'top_left'
                        index_reset = False
                    else:
                        elapsed_time = time.time() - start_time_left
                        if elapsed_time >= 2:  # Check if finger is in corner for 3 seconds
                            if is_recording:
                                out.release()
                                if last_saved_video != f'{out_prefix}put{recording_count - 1}.mp4':
                                    print(f'Saved recording {out_prefix}put{recording_count - 1}.mp4')
                                    last_saved_video = f'{out_prefix}put{recording_count - 1}.mp4'
                                else:
                                    os.remove(os.path.join(output_dir, last_saved_video))
                                    print(f'Discarded recording {last_saved_video}')
                                no_hand_count = 0
                                break  # Exit the loop
                else:
                    stop_timer_left = False

                if not is_recording:
                    is_recording = True
                    output_path = os.path.join(output_dir, f'{out_prefix}put{recording_count}.mp4')
                    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame.shape[1], frame.shape[0]))
                    print(f'Started recording {output_path}')
                    recording_count += 1
                    last_saved_video = None  # Reset the last saved video tracker

            elif is_recording:
                no_hand_count += 1
                if no_hand_count >= recording_delay:
                    is_recording = False
                    out.release()
                    print(f'Stopped recording {out_prefix}put{recording_count - 1}.mp4')
                    no_hand_count = 0

            if is_recording:
                out.write(frame)

            cv2.imshow('Hand Tracking', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                if is_recording:
                    out.release()
                break

        cap.release()
        cv2.destroyAllWindows()


def count_videos_per_prefix(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Create a dictionary to store the video counts per prefix
    video_counts = {}

    # Count videos per prefix
    for filename in files:
        if filename.endswith('.mp4'):
            prefix = filename.split('put')[0]
            video_counts[prefix] = video_counts.get(prefix, 0) + 1

    # Create a list of video counts, ordered by prefix
    video_counts_list = [video_counts.get(prefix, 0) for prefix in sorted(set(video_counts.keys()))]

    return video_counts_list


def record_delete(dir_vid):

    #record(dir_vid)
    delete_last_index_files(dir_vid)
    video_counts_list = count_videos_per_prefix(dir_vid)
    return video_counts_list

if __name__ == "__main__":
    dir_vid = "/home/tester/finalProject/videos"
    record_delete(dir_vid)
