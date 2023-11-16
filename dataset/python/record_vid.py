import cv2
import mediapipe as mp
import os
import time  # Import the time module

def record(output_dir):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.75) as hands:
        is_recording = False
        recording_count = 0
        no_hand_count = 0
        recording_delay = 20

        os.makedirs(output_dir, exist_ok=True)

        # Initialize timer variables
        start_time = None
        stop_timer = False

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

                if index_finger_tip_landmark.x * width < corner_boundary and index_finger_tip_landmark.y * height < corner_boundary:
                    if not stop_timer:
                        start_time = time.time()
                        stop_timer = True
                    else:
                        elapsed_time = time.time() - start_time
                        if elapsed_time >= 2:  # Check if finger is in corner for 3 seconds
                            if is_recording:
                                is_recording = False
                                out.release()
                                print(f'Stopped recording output{recording_count}.mp4')
                                no_hand_count = 0
                            break
                else:
                    stop_timer = False

                if not is_recording:
                    is_recording = True
                    recording_count += 1
                    output_path = os.path.join(output_dir, f'output{recording_count}.mp4')
                    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame.shape[1], frame.shape[0]))
                    print(f'Started recording {output_path}')

            elif is_recording:
                no_hand_count += 1
                if no_hand_count >= recording_delay:
                    is_recording = False
                    out.release()
                    print(f'Stopped recording output{recording_count}.mp4')
                    no_hand_count = 0

            if is_recording:
                out.write(frame)

            cv2.imshow('Hand Tracking', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    output_dir = '/home/tester/finalProject/videos'
    record(output_dir)
