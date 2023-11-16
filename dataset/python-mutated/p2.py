from face_detector import *
DIR_PATH = '/home/deeplearning/Desktop/Attendance-System/Images/video_dataset3'
bins = [{p} for p in people]
Q = []

def get_bin_score(fid, bid):
    if False:
        i = 10
        return i + 15
    return sum([sim_mat[fid][b_bid] for b_bid in bins[bid]]) / len(bins[bid])

def get_best_score(fid):
    if False:
        for i in range(10):
            print('nop')
    best_score = -1.0
    best_bin = 0
    arr = faces[fid].bin_score
    for bid in range(len(arr)):
        if arr[bid] > best_score:
            best_score = arr[bid]
            best_bin = bid
    (faces[fid].best_score, faces[fid].best_bin) = (best_score, best_bin)

def update_bin_score(fid, bid, new_fid):
    if False:
        while True:
            i = 10
    old_val = faces[fid].bin_score[bid]
    s = len(bins[bid]) - 1
    new_val = (old_val * s + sim_mat[fid][new_fid]) / (s + 1)
    faces[fid].bin_score[bid] = new_val
    if new_val > faces[fid].best_score:
        faces[fid].best_score = new_val
        faces[fid].best_bin = bid

def best_score_face():
    if False:
        print('Hello World!')
    (best_fid, best_score) = (-1, -1)
    for fid in Q:
        if faces[fid].best_score > best_score:
            (best_fid, best_score) = (fid, faces[fid].best_score)
    return best_fid
for fid in range(len(faces)):
    if fid not in people:
        Q.append(fid)
for fid in Q:
    faces[fid].bin_score = [get_bin_score(fid, bid) for bid in range(len(bins))]
    get_best_score(fid)
while len(Q) > 0:
    best_fid = best_score_face()
    Q.remove(best_fid)
    bid = faces[best_fid].best_bin
    bins[bid].add(best_fid)
    for fid in Q:
        update_bin_score(fid, bid, best_fid)

def writeDataset():
    if False:
        print('Hello World!')
    if not os.path.exists(DIR_PATH):
        os.mkdir(DIR_PATH)
    for (bin_num, b) in enumerate(bins):
        BIN_PATH = DIR_PATH + '/bin' + str(bin_num) + '/'
        if not os.path.exists(BIN_PATH):
            os.mkdir(BIN_PATH)
        for fid in b:
            IMAGE_PATH = BIN_PATH + faces[fid].img_name + '_' + str(faces[fid].box_no) + '.jpg'
            cv2.imwrite(IMAGE_PATH, faces[fid].myFace)
print(bins)
writeDataset()