import cv2

def record_video(output_file, width=640, height=480):
    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4 format
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (width, height))

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            # Write the frame to the output video file
            out.write(frame)

            # Display the frame
            cv2.imshow('Recording', frame)

        # Check for user interruption
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and writer objects
    cap.release()
    out.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    output_file = "/home/tester/finalProject/mode2_videos/output.mp4"  # Output video file name with .mp4 extension
    record_video(output_file)
