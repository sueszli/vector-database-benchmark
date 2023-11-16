## P2

# All class definitions are defined in this module
#from p1 import *
from face_detector import *

#Input
#faces = []
#sim_mat = []
#people = []

DIR_PATH = '/home/deeplearning/Desktop/Attendance-System/Images/video_dataset3'

# Declaring Data
bins = [{p} for p in people]
Q = []


# Defining Methods

# for every face in the bin, sum the similarity
# and take average
def get_bin_score(fid,bid):
    return sum([sim_mat[fid][b_bid] for b_bid in bins[bid]])/len(bins[bid])

# best bin score is the bin with the max confidence
def get_best_score(fid):
    best_score = -1.0
    best_bin = 0
    arr = faces[fid].bin_score
    for bid in range(len(arr)):
        if arr[bid] > best_score:
            best_score = arr[bid]
            best_bin = bid
    faces[fid].best_score,faces[fid].best_bin = best_score, best_bin

# For that bid which got updated, for every face
# update the score corresponding to that bin
# We can just retake the average by doing
# old_score * n + new_val / (n+1)
# Finally we update best score and best bid
def update_bin_score(fid, bid, new_fid):
    old_val = faces[fid].bin_score[bid]
    s = len(bins[bid]) - 1
    new_val = (old_val*s + sim_mat[fid][new_fid])/(s+1)
    faces[fid].bin_score[bid] = new_val
    if new_val > faces[fid].best_score:
        faces[fid].best_score = new_val
        faces[fid].best_bin = bid

# iterate over all faces and get the face with highest
# maximum bin_score
def best_score_face():
    best_fid,best_score = -1,-1
    for fid in Q:
        if faces[fid].best_score > best_score:
            best_fid,best_score = fid,faces[fid].best_score
    return best_fid


# Herein lies the puny algorithm
for fid in range(len(faces)):
    if fid not in people:
        Q.append(fid)

# Impure method, gets you best_bin and best_score for
# every face
for fid in Q:
    faces[fid].bin_score = [get_bin_score(fid,bid) for bid in range(len(bins))]
    get_best_score(fid)

while len(Q)>0:
    best_fid = best_score_face()
    Q.remove(best_fid)
    bid = faces[best_fid].best_bin
    bins[bid].add(best_fid)
    for fid in Q:
        update_bin_score(fid,bid,best_fid)

def writeDataset():
    if not os.path.exists(DIR_PATH):
        os.mkdir(DIR_PATH)
    for bin_num, b in enumerate(bins):
        BIN_PATH = DIR_PATH + '/bin' + str(bin_num) + '/'
        if not os.path.exists(BIN_PATH):
            os.mkdir(BIN_PATH)
        for fid in b:
            #if fid in people:
                #image_path = BASE_IMAGE_DIR
            #else:
            #image_path = IMAGE_BASE_PATH
            #img = fr.load_image_file(image_path + '/' + faces[fid].img_name)
            #x1, y2, x2, y1 = faces[fid].bound_box
            #img = img[x1: x2, y1: y2]
            IMAGE_PATH = BIN_PATH + faces[fid].img_name + '_' + str(faces[fid].box_no) + '.jpg'
            cv2.imwrite(IMAGE_PATH, faces[fid].myFace)

print(bins)
writeDataset()
