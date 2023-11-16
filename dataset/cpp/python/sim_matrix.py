import numpy as np
import matplotlib.pyplot as plt

def matrix_norm(W):
    '''归一化'''
    mx = np.max(np.max(W, axis=1))
    mn = np.min(np.min(W, axis=1))
    for i in range(len(W)):
        for j in range((len(W))):
            W[i, j] = (W[i, j] - mn) / (mx - mn)
    return W
    
def cos_sim(vec1, vec2):
    vector_a = np.mat(vec1)
    vector_b = np.mat(vec2)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    sim = num / denom
    return sim

def show_matrix_features(W):
    # ls = []
    # for line in W:
    #     for num in line:
    #         ls.append(num)
    # ls.sort(reverse=True)
    # ls = np.array(ls)
    # fig = plt.figure()
    ax = plt.subplot()
    im = ax.imshow(W, cmap=plt.cm.hot_r)
    ax.set_xticks(np.arange(len(W)))
    ax.set_yticks(np.arange(len(W)))
    plt.colorbar(im)
    plt.show()

class similarity_matrix:
    def __init__(self, dl):
        self.dl = dl
        self.label_similarity_matrix = self.get_label_similarity()
        self.label_based_similartiy_matrix = self.getLabelBasedItemSimilarity()
        self.image_based_similarity_matrix = self.getImageBasedItemSimilarity(cos_sim)
        self.collaborative_similarity_matrix = self.getCollaboritiveSimilarity()
        # show_matrix_features(self.collaborative_similarity_matrix)

    def get_label_similarity(self):
        labels_size = self.dl.__labelSize__()
        labels_list = self.dl.labels_list
        labels2idx = self.dl.label2index
        labels_byItem = self.dl.labels_byItem
        label_freq = self.dl.label_freq
        W = np.zeros((labels_size, labels_size),dtype=np.float16)
        '''统计共同出现的次数'''
        for line in labels_byItem:
            for i in range(len(line)):
                label_i = line[i]
                i_i = labels2idx[label_i]
                for j in range(len(line)):
                    label_j = line[j]
                    i_j = labels2idx[label_j]
                    if i_i != i_j:
                        W[i_i][i_j] += 1
        '''求相似度'''
        for i in range(len(W)):
            for j in range(len(W)):
                if i == j:
                    W[i, j] = 1
                    continue
                label_i_num = label_freq[labels_list[i]]
                label_j_num = label_freq[labels_list[j]]
                W[i, j] /= np.sqrt(label_i_num * label_j_num)
        W = matrix_norm(W)
        return W

    def getLabelBasedItemSimilarity(self):
        itemSize = self.dl.__len__()
        W = np.zeros((itemSize, itemSize),dtype=np.float16)
        
        for i in range(len(W)):
            for j in range(len(W)):
                '''获取两个item的label list'''
                labels_i = self.dl.getLabels(i)
                labels_j = self.dl.getLabels(j)
                if i == j:
                    continue
                W[i, j] += self.cal_label_list_similarity(labels_i, labels_j)
        W = matrix_norm(W)
        return W
    
    def getImageBasedItemSimilarity(self, sim_func):
        itemSize = self.dl.__len__()
        W = np.zeros((itemSize, itemSize),dtype=np.float16)
        
        for i in range(len(W)):
            for j in range(len(W)):
                img_i = self.dl.items_list[i]
                img_j = self.dl.items_list[j]
                img_fm_i = self.dl.img_featuremaps[img_i]
                img_fm_j = self.dl.img_featuremaps[img_j]
                if i == j:
                    continue
                W[i, j] = sim_func(img_fm_i, img_fm_j)
        W = matrix_norm(W)
        return W
    
    def getCollaboritiveSimilarity(self):
        clb_sim_mat = self.label_based_similartiy_matrix + self.image_based_similarity_matrix
        clb_sim_mat = matrix_norm(clb_sim_mat)
        return clb_sim_mat
    
    def cal_label_list_similarity(self, ls1, ls2):
        score = 0
        counter = 0
        lb_mat = self.label_similarity_matrix
        for lb_i in ls1:
            idx_i = self.dl.label2index[lb_i]
            mx = 0
            for lb_j in ls2:
                idx_j = self.dl.label2index[lb_j]
                mx = max(lb_mat[idx_i, idx_j], mx)
            score += mx
            counter += 1
        return score / counter
    
# dl = dataLoader('./filelist/filelist.csv', './imgs', './features/features.json')
# sm = similarity_matrix(dl)