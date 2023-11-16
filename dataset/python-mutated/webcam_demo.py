from cvs import *
print('import tensorflow...wait...')
import tensorflow as tf
import time
import argparse
import posenet
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=50)
parser.add_argument('--cam', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=640)
parser.add_argument('--cam_height', type=int, default=480)
parser.add_argument('--scale_factor', type=float, default=0.7125)
args = parser.parse_args()
cam_id = args.cam

def main():
    if False:
        i = 10
        return i + 15
    cap = cvs.VideoCapture(0)
    with tf.Session() as sess:
        print('load models...')
        (model_cfg, model_outputs) = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']
        start = time.time()
        frame_count = 0
        while True:
            sleep(30)
            img = cvs.read()
            frame_count += 1
            if img is None:
                continue
            if cam_id > 0:
                img = cvs.flip(img, 0)
            (input_image, display_image, output_scale) = posenet.read_cap(img, scale_factor=args.scale_factor, output_stride=output_stride)
            (heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result) = sess.run(model_outputs, feed_dict={'image:0': input_image})
            (pose_scores, keypoint_scores, keypoint_coords) = posenet.decode_multi.decode_multiple_poses(heatmaps_result.squeeze(axis=0), offsets_result.squeeze(axis=0), displacement_fwd_result.squeeze(axis=0), displacement_bwd_result.squeeze(axis=0), output_stride=output_stride, max_pose_detections=10, min_pose_score=0.15)
            keypoint_coords *= output_scale
            overlay_image = posenet.draw_skel_and_kp(display_image, pose_scores, keypoint_scores, keypoint_coords, min_pose_score=0.15, min_part_score=0.1)
            cvs.imshow(overlay_image)
            frame_count += 1
            lbs = 'Average FPS: ' + str(frame_count / (time.time() - start))
            cvs.setLbs(lbs)

class MyApp(App):

    def __init__(self, *args):
        if False:
            return 10
        super(MyApp, self).__init__(*args)

    def idle(self):
        if False:
            return 10
        self.lbl.set_text(cvs.getLbs())
        pass

    def main(self):
        if False:
            return 10
        main_container = gui.VBox(width=360, height=680, style={'margin': '0px auto'})
        self.aidcam = OpencvVideoWidget(self, width=340, height=480)
        self.aidcam.style['margin'] = '10px'
        self.aidcam.set_identifier('myimage_receiver')
        main_container.append(self.aidcam)
        self.lbl = gui.Label('This show FPS!', width=360, height=30, margin='10px')
        main_container.append(self.lbl)
        return main_container
if __name__ == '__main__':
    initcv(main)
    startcv(MyApp)