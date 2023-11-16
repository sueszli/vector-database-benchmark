import numpy as np
import time
import torch
from .re_ranking import re_ranking

def evaluate_with_query_set(gallery_labels, gallery_groups, gallery_features, query_labels, query_groups, query_features, use_rerank=False, rerank_k1=20, rerank_k2=6, rerank_lambda=0.3, is_market1501=False):
    if False:
        print('Hello World!')
    ap = 0.0
    CMC = torch.IntTensor(len(gallery_labels)).zero_()
    q_g_dist = np.dot(query_features, np.transpose(gallery_features))
    if use_rerank:
        print('Calculate re-ranked distances..')
        q_q_dist = np.dot(query_features, np.transpose(query_features))
        g_g_dist = np.dot(gallery_features, np.transpose(gallery_features))
        since = time.time()
        distances = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=rerank_k1, k2=rerank_k2, lambda_value=rerank_lambda)
        time_elapsed = time.time() - since
        print('Reranking complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    else:
        distances = -q_g_dist
    norm = 0
    skip = 1
    for i in range(len(query_labels))[::skip]:
        (ap_tmp, CMC_tmp) = evaluate_helper(distances[i, :], query_labels[i], query_groups[i], gallery_labels, gallery_groups, is_market1501)
        if CMC_tmp[0] == -1:
            continue
        norm += 1
        ap += ap_tmp
        CMC = CMC + CMC_tmp
    if is_market1501:
        norm = len(query_labels) / float(skip)
    ap = ap / norm
    CMC = CMC.float()
    CMC = CMC / norm
    print('Rank@1:{:.1f}, rank@5:{:.1f}, mAP:{:.2f}'.format(100 * CMC[0], 100 * CMC[4], ap))
    return (CMC, ap)

def evaluate_helper(score, ql, qc, gl, gc, is_market1501=False):
    if False:
        return 10
    assert type(gl) == np.ndarray, 'Input gl has to be a numpy ndarray'
    assert type(gc) == np.ndarray, 'Input gc has to be a numpy ndarray'
    index = np.argsort(score)
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index2 = np.intersect1d(query_index, camera_index)
    if is_market1501:
        junk_index1a = np.argwhere(gl == -1)
        junk_index1b = np.argwhere(gl == '-1')
        junk_index1 = np.append(junk_index1a, junk_index1b)
        junk_index = np.append(junk_index2, junk_index1)
    else:
        junk_index = junk_index2
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp

def compute_mAP(index, good_index, junk_index):
    if False:
        while True:
            i = 10
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:
        cmc[0] = -1
        return (ap, cmc)
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask)
    rows_good = rows_good.flatten()
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2
    return (ap, cmc)