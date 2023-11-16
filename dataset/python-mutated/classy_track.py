import sys
sys.path.insert(0, './sort')
import numpy as np
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from utils.datasets import LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
import skimage
from sort import *
torch.set_printoptions(precision=3)
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def bbox_rel(*xyxy):
    if False:
        while True:
            i = 10
    '" Calculates the relative bounding box from absolute pixel values. '
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = bbox_left + bbox_w / 2
    y_c = bbox_top + bbox_h / 2
    w = bbox_w
    h = bbox_h
    return (x_c, y_c, w, h)

def compute_color_for_labels(label):
    if False:
        while True:
            i = 10
    '\n    Simple function that adds fixed color depending on the class\n    '
    color = [int(p * (label ** 2 - label + 1) % 255) for p in palette]
    return tuple(color)

def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0)):
    if False:
        i = 10
        return i + 15
    for (i, box) in enumerate(bbox):
        (x1, y1, x2, y2) = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = f'{names[cat]} | {id}'
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

def detect(opt, *args):
    if False:
        return 10
    (out, source, weights, view_img, save_txt, imgsz, save_img, sort_max_age, sort_min_hits, sort_iou_thresh) = (opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.save_img, opt.sort_max_age, opt.sort_min_hits, opt.sort_iou_thresh)
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    sort_tracker = Sort(max_age=sort_max_age, min_hits=sort_min_hits, iou_threshold=sort_iou_thresh)
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    half = device.type != 'cpu'
    model = torch.load(weights, map_location=device)['model'].float()
    model.to(device).eval()
    if half:
        model.half()
    (vid_path, vid_writer) = (None, None)
    if webcam:
        view_img = True
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        dataset = LoadImages(source, img_size=imgsz)
    names = model.module.names if hasattr(model, 'module') else model.names
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)
    _ = model(img.half() if half else img) if device.type != 'cpu' else None
    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'
    for (frame_idx, (path, img, im0s, vid_cap)) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img, augment=opt.augment)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        for (i, det) in enumerate(pred):
            if webcam:
                (p, s, im0) = (path[i], '%g: ' % i, im0s[i].copy())
            else:
                (p, s, im0) = (path, '', im0s)
            s += f'{img.shape[2:]}'
            save_path = str(Path(out) / Path(p).name)
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
                s += f' - {n} {names[int(c)]}'
            dets_to_sort = np.empty((0, 6))
            for (x1, y1, x2, y2, conf, detclass) in det.cpu().detach().numpy():
                dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))
            print('\n')
            print('Input into SORT:\n', dets_to_sort, '\n')
            tracked_dets = sort_tracker.update(dets_to_sort)
            print('Output from SORT:\n', tracked_dets, '\n')
            if len(tracked_dets) > 0:
                bbox_xyxy = tracked_dets[:, :4]
                identities = tracked_dets[:, 8]
                categories = tracked_dets[:, 4]
                draw_boxes(im0, bbox_xyxy, identities, categories, names)
            if save_txt and len(tracked_dets) != 0:
                for (j, tracked_dets) in enumerate(tracked_dets):
                    bbox_x1 = tracked_dets[0]
                    bbox_y1 = tracked_dets[1]
                    bbox_x2 = tracked_dets[2]
                    bbox_y2 = tracked_dets[3]
                    category = tracked_dets[4]
                    u_overdot = tracked_dets[5]
                    v_overdot = tracked_dets[6]
                    s_overdot = tracked_dets[7]
                    identity = tracked_dets[8]
                    with open(txt_path, 'a') as f:
                        f.write(f'{frame_idx},{bbox_x1},{bbox_y1},{bbox_x2},{bbox_y2},{category},{u_overdot},{v_overdot},{s_overdot},{identity}\n')
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):
                    raise StopIteration
            if save_img:
                print('saving img!')
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    print('saving video!')
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)
    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':
            os.system('open ' + save_path)
    print('Done. (%.3fs)' % (time.time() - t0))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='best.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='inference/images', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')
    parser.add_argument('--img-size', type=int, default=1080, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-img', action='store_true', help='save video file to output folder (disable for speed)')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, default=[i for i in range(5)], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--sort-max-age', type=int, default=5, help='keep track of object even if object is occluded or not detected in n frames')
    parser.add_argument('--sort-min-hits', type=int, default=2, help='start tracking only after n number of objects detected')
    parser.add_argument('--sort-iou-thresh', type=float, default=0.2, help='intersection-over-union threshold between two frames for association')
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)
    with torch.no_grad():
        detect(args)