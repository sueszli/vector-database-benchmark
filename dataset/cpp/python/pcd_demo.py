from argparse import ArgumentParser

from mmdet3d.apis import inference_detector, init_model, show_result_meshlab
import numpy as np
from math import pi
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


def draw_bbox_cv2(image, bboxes):
    for bbox in bboxes:
        x, y, _, w, h, _, angle = bbox
        print(x, y, w, h)
        rect = ((x.item(), y.item()), (w.item(), h.item()), angle.item()/5)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
    return image

def load_from_bin(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return obj

def normalize_depth(val, min_v, max_v):
    """ 
    print 'normalized depth value' 
    normalize values to 0-255 & close distance value has high value. (similar to stereo vision's disparity map)
    """
    return (((max_v - val) / (max_v - min_v)) * 255).astype(np.uint8)

def in_range_points(points, x, y, z, x_range, y_range, z_range):
    """ extract in-range points """
    return points[np.logical_and.reduce((x > x_range[0], x < x_range[1], y > y_range[0], \
                                         y < y_range[1], z > z_range[0], z < z_range[1]))]

def velo_points_2_top_view(points, x_range, y_range, z_range, scale):
    
    # Projecting to 2D
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    dist = np.sqrt(x ** 2 + y ** 2)
    
    # extract in-range points
    x_lim = in_range_points(x, x, y, z, x_range, y_range, z_range)
    y_lim = in_range_points(y, x, y, z, x_range, y_range, z_range)
    dist_lim = in_range_points(dist, x, y, z, x_range, y_range, z_range)
    
    # * x,y,z range are based on lidar coordinates
    x_size = int((y_range[1] - y_range[0]))
    y_size = int((x_range[1] - x_range[0]))
    
    # convert 3D lidar coordinates(vehicle coordinates) to 2D image coordinates
    # Velodyne coordinates info : http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
    # scale - for high resolution
    x_img = -(y_lim * scale).astype(np.int32)
    y_img = -(x_lim * scale).astype(np.int32)

    # shift negative points to positive points (shift minimum value to 0)
    x_img += int(np.trunc(y_range[1] * scale))
    y_img += int(np.trunc(x_range[1] * scale))

    # normalize distance value & convert to depth map
    max_dist = np.sqrt((max(x_range)**2) + (max(y_range)**2))
    dist_lim = normalize_depth(dist_lim, min_v=0, max_v=max_dist)
    print(dist_lim)
    # array to img
    img = np.zeros([y_size * scale + 1, x_size * scale + 1], dtype=np.uint8)
    img[y_img, x_img] = dist_lim
    
    return img




def main():
    parser = ArgumentParser()
    parser.add_argument('pcd', default='data/custom/velo/0.bin', help='Point cloud file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.0, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()
    
    pcd_paths = [args.pcd]
    
    for pcd_path in pcd_paths:
        model = init_model(args.config, args.checkpoint, device=args.device)
        # test a single image



        front_result, _ = inference_detector(model, pcd_path, is_front=True)
        back_result, _ = inference_detector(model, pcd_path, is_front=False)

        # show the results
        length = front_result[0]['scores_3d'].shape[0]
        velo_points = load_from_bin(pcd_path)
        top_image = velo_points_2_top_view(velo_points, x_range=(-48, 60), y_range=(-96, 96), z_range=(-4, 4), scale=10)
        plt.imshow(top_image, cmap='gray')
        colors = ['red', 'blue', 'green']
        for i in range(length):
            if front_result[0]['scores_3d'][i] < 0.4:
                continue
            y, x, z, w, l, h, rw = front_result[0]['boxes_3d'].tensor[i] * 10
            label = front_result[0]['labels_3d'][i]
            ego_size_x = l
            ego_size_y = w
            ego_x = 960 - x - l/2
            ego_y = 600 - y - w/2
            ego_angle = -1*rw * 18 / pi

            rec = patches.Rectangle((ego_x,ego_y), ego_size_x, ego_size_y,  angle=ego_angle, rotation_point='center', color=colors[label], fill=False)
            plt.gca().add_patch(rec)

#         back file
        length = back_result[0]['scores_3d'].shape[0]
        for i in range(length):
            if back_result[0]['scores_3d'][i] < 0.4:
                continue

            y, x, z, w, l, h, rw = back_result[0]['boxes_3d'].tensor[i] * 10
            label = back_result[0]['labels_3d'][i]
            ego_size_x = l
            ego_size_y = w
            ego_x = 960 - x - l/2
            ego_y = 600 + y - w/2
            ego_angle = rw * 18 / pi

            rec = patches.Rectangle((ego_x,ego_y), ego_size_x, ego_size_y,  angle=ego_angle, rotation_point='center', color=colors[label], fill=False)
            plt.gca().add_patch(rec)



        plt.axis("off")
        matplotlib.pyplot.savefig("results/{}.png".format(pcd_path.split('/')[-1][:-4]), dpi=387, pad_inches=0, bbox_inches='tight')
        plt.close()
    
if __name__ == '__main__':
    main()