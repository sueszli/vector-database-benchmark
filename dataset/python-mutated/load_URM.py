import pandas as pd
import scipy.sparse as sps

def load_URM(file_path):
    if False:
        i = 10
        return i + 15
    data = pd.read_csv(file_path)
    user_list = data['row'].tolist()
    item_list = data['col'].tolist()
    rating_list = data['data'].tolist()
    return sps.coo_matrix((rating_list, (user_list, item_list))).tocsr()