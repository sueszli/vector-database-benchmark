from __future__ import print_function
from builtins import range
import sys, os
from cntk_helpers import *
import scipy.sparse
import scipy.io as sio
import pickle as cp
import numpy as np
import fastRCNN

class imdb_data(fastRCNN.imdb):

    def __init__(self, image_set, classes, maxNrRois, imgDir, roiDir, cacheDir, boAddGroundTruthRois):
        if False:
            i = 10
            return i + 15
        fastRCNN.imdb.__init__(self, image_set + '.cache')
        self._image_set = image_set
        self._maxNrRois = maxNrRois
        self._imgDir = imgDir
        self._roiDir = roiDir
        self._cacheDir = cacheDir
        self._imgSubdirs = {'train': ['positive', 'negative'], 'test': ['testImages']}
        self._classes = classes
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._image_ext = '.jpg'
        (self._image_index, self._image_subdirs) = self._load_image_set_index()
        self._roidb_handler = self.selective_search_roidb
        self._boAddGroundTruthRois = boAddGroundTruthRois

    @property
    def cache_path(self):
        if False:
            for i in range(10):
                print('nop')
        return self._cacheDir

    def image_path_at(self, i):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the absolute path to image i in the image sequence.\n        '
        return self.image_path_from_index(self._image_subdirs[i], self._image_index[i])

    def image_path_from_index(self, subdir, fname):
        if False:
            i = 10
            return i + 15
        '\n        Construct an image path from the image\'s "index" identifier.\n        '
        image_path = os.path.join(self._imgDir, subdir, fname)
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        if False:
            while True:
                i = 10
        '\n        Compile list of image indices and the subdirectories they are in.\n        '
        image_index = []
        image_subdirs = []
        for subdir in self._imgSubdirs[self._image_set]:
            imgFilenames = getFilesInDirectory(os.path.join(self._imgDir, subdir), self._image_ext)
            image_index += imgFilenames
            image_subdirs += [subdir] * len(imgFilenames)
        return (image_index, image_subdirs)

    def gt_roidb(self):
        if False:
            return 10
        '\n        Return the database of ground-truth regions of interest.\n\n        This function loads/saves from/to a cache file to speed up future calls.\n        '
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cp.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb
        gt_roidb = [self._load_annotation(i) for i in range(self.num_images)]
        with open(cache_file, 'wb') as fid:
            cp.dump(gt_roidb, fid, cp.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb

    def selective_search_roidb(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the database of selective search regions of interest.\n        Ground-truth ROIs are also included.\n\n        This function loads/saves from/to a cache file to speed up future calls.\n        '
        cache_file = os.path.join(self.cache_path, self.name + '_selective_search_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                if sys.version_info[0] < 3:
                    roidb = cp.load(fid)
                else:
                    roidb = cp.load(fid, encoding='latin1')
            print('{} ss roidb loaded from {}'.format(self.name, cache_file))
            return roidb
        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_roidb(gt_roidb)
        if self._boAddGroundTruthRois:
            roidb = self.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = ss_roidb
        if self._maxNrRois and self._maxNrRois > 0:
            print('Only keeping the first %d ROIs..' % self._maxNrRois)
            for i in range(self.num_images):
                gt_overlaps = roidb[i]['gt_overlaps']
                gt_overlaps = gt_overlaps.todense()[:self._maxNrRois]
                gt_overlaps = scipy.sparse.csr_matrix(gt_overlaps)
                roidb[i]['gt_overlaps'] = gt_overlaps
                roidb[i]['boxes'] = roidb[i]['boxes'][:self._maxNrRois, :]
                roidb[i]['gt_classes'] = roidb[i]['gt_classes'][:self._maxNrRois]
        with open(cache_file, 'wb') as fid:
            cp.dump(roidb, fid, cp.HIGHEST_PROTOCOL)
        print('wrote ss roidb to {}'.format(cache_file))
        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        if False:
            for i in range(10):
                print('nop')
        box_list = []
        for (imgFilename, subdir) in zip(self._image_index, self._image_subdirs):
            roiPath = '{}/{}/{}.roi.txt'.format(self._roiDir, subdir, imgFilename[:-4])
            assert os.path.exists(roiPath), 'Error: rois file not found: ' + roiPath
            rois = np.loadtxt(roiPath, np.int32)
            box_list.append(rois)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_annotation(self, imgIndex):
        if False:
            for i in range(10):
                print('nop')
        '\n        Load image and bounding boxes info from human annotations.\n        '
        imgPath = self.image_path_at(imgIndex)
        bboxesPaths = imgPath[:-4] + '.bboxes.tsv'
        labelsPaths = imgPath[:-4] + '.bboxes.labels.tsv'
        if not os.path.exists(bboxesPaths) or not os.path.exists(labelsPaths):
            return None
        bboxes = np.loadtxt(bboxesPaths, np.float32)
        if len(bboxes.shape) == 1:
            bboxes = np.array([bboxes])
        labels = readFile(labelsPaths)
        indicesToKeep = find(labels, lambda x: x != 'EXCLUDE' and x != 'UNDECIDED')
        bboxes = [bboxes[i] for i in indicesToKeep]
        labels = [labels[i] for i in indicesToKeep]
        num_objs = len(bboxes)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros(num_objs, dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        for (bboxIndex, (bbox, label)) in enumerate(zip(bboxes, labels)):
            cls = self._class_to_ind[label.decode('utf-8')]
            boxes[bboxIndex, :] = bbox
            gt_classes[bboxIndex] = cls
            overlaps[bboxIndex, cls] = 1.0
        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'boxes': boxes, 'gt_classes': gt_classes, 'gt_overlaps': overlaps, 'flipped': False}

    def evaluate_detections(self, all_boxes, output_dir, use_07_metric=False):
        if False:
            i = 10
            return i + 15
        aps = []
        for (classIndex, className) in enumerate(self._classes):
            if className != '__background__':
                (rec, prec, ap) = self._evaluate_detections(classIndex, all_boxes, use_07_metric=use_07_metric)
                aps += [ap]
                print('AP for {:>15} = {:.4f}'.format(className, ap))
        print('Mean AP = {:.4f}'.format(np.nanmean(aps)))

    def _evaluate_detections(self, classIndex, all_boxes, overlapThreshold=0.5, use_07_metric=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Top level function that does the PASCAL VOC evaluation.\n\n        [overlapThreshold]: Overlap threshold (default = 0.5)\n        [use_07_metric]: Whether to use VOC07's 11 point AP computation (default False)\n        "
        assert len(all_boxes) == self.num_classes
        assert len(all_boxes[0]) == self.num_images
        gtInfos = []
        for imgIndex in range(self.num_images):
            imgPath = self.image_path_at(imgIndex)
            imgSubir = os.path.normpath(imgPath).split(os.path.sep)[-2]
            bboxesPaths = imgPath[:-4] + '.bboxes.tsv'
            labelsPaths = imgPath[:-4] + '.bboxes.labels.tsv'
            if os.path.exists(bboxesPaths) and os.path.exists(labelsPaths):
                (gtBoxes, gtLabels) = readGtAnnotation(imgPath)
                gtBoxes = [box for (box, label) in zip(gtBoxes, gtLabels) if label.decode('utf-8') == self.classes[classIndex]]
            else:
                gtBoxes = []
            gtInfos.append({'bbox': np.array(gtBoxes), 'difficult': [False] * len(gtBoxes), 'det': [False] * len(gtBoxes)})
        detBboxes = []
        detImgIndices = []
        detConfidences = []
        for imgIndex in range(self.num_images):
            dets = all_boxes[classIndex][imgIndex]
            if dets != []:
                for k in range(dets.shape[0]):
                    detImgIndices.append(imgIndex)
                    detConfidences.append(dets[k, -1])
                    detBboxes.append([dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1])
        detBboxes = np.array(detBboxes)
        detConfidences = np.array(detConfidences)
        (rec, prec, ap) = self._voc_computePrecisionRecallAp(class_recs=gtInfos, confidence=detConfidences, image_ids=detImgIndices, BB=detBboxes, ovthresh=overlapThreshold, use_07_metric=use_07_metric)
        return (rec, prec, ap)

    def _voc_computePrecisionRecallAp(self, class_recs, confidence, image_ids, BB, ovthresh=0.5, use_07_metric=False):
        if False:
            while True:
                i = 10
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
                ih = np.maximum(iymax - iymin + 1.0, 0.0)
                inters = iw * ih
                uni = (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0) + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0) - inters
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.0
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.0
            else:
                fp[d] = 1.0
        npos = sum([len(cr['bbox']) for cr in class_recs])
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = computeAveragePrecision(rec, prec, use_07_metric)
        return (rec, prec, ap)