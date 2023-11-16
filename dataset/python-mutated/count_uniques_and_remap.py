import numpy as np
import pandas as pd

def count_uniques_and_remap(urm: pd.DataFrame):
    if False:
        for i in range(10):
            print('nop')
    unique_users = urm.row.unique()
    unique_items = urm.col.unique()
    (num_users, min_user_id, max_user_id) = (unique_users.size, unique_users.min(), unique_users.max())
    (num_items, min_item_id, max_item_id) = (unique_items.size, unique_items.min(), unique_items.max())
    n_interactions = len(urm)
    print('Number of items\t {}, Number of users\t {}'.format(num_items, num_users))
    print('Max ID items\t {}, Max Id users\t {}\n'.format(max(unique_items), max(unique_users)))
    print('Average interactions per user {:.2f}'.format(n_interactions / num_users))
    print('Average interactions per item {:.2f}\n'.format(n_interactions / num_items))
    print('Sparsity {:.2f} %'.format((1 - float(n_interactions) / (num_items * num_users)) * 100))
    mapping_user_id = pd.DataFrame({'mapped_row': np.arange(num_users), 'row': unique_users})
    mapping_item_id = pd.DataFrame({'mapped_col': np.arange(num_items), 'col': unique_items})
    urm = pd.merge(left=urm, right=mapping_user_id, how='inner', on='row')
    urm = pd.merge(left=urm, right=mapping_item_id, how='inner', on='col')
    return urm