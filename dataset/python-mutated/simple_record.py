import cv2

def record_video(output_file, width=640, height=480):
    if False:
        print('Hello World!')
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (width, height))
    while True:
        (ret, frame) = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow('Recording', frame)
        if cv2.waitKey(1) & 255 == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    output_file = '/home/tester/finalProject/mode2_videos/output.mp4'
    record_video(output_file)