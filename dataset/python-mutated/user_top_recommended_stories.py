import numpy as np
from dagster import asset
from pandas import DataFrame
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from sklearn.decomposition import TruncatedSVD
from .user_story_matrix import IndexedCooMatrix

@asset(io_manager_key='warehouse_io_manager', key_prefix=['snowflake', 'recommender'])
def user_top_recommended_stories(context, recommender_model: TruncatedSVD, user_story_matrix: IndexedCooMatrix) -> DataFrame:
    if False:
        return 10
    'The top stories for each commenter (user).'
    XV = recommender_model.transform(user_story_matrix.matrix)
    XV[np.abs(XV) < 1] = 0
    sparse_XV = csr_matrix(XV)
    context.log.info(f'sparse_XV shape: {sparse_XV.shape}')
    context.log.info(f'sparse_XV non-zero: {sparse_XV.count_nonzero()}')
    recommender_model.components_[np.abs(recommender_model.components_) < 0.01] = 0
    sparse_components = csc_matrix(recommender_model.components_)
    context.log.info(f'recommender_model.components_ shape: {recommender_model.components_.shape}')
    context.log.info(f'sparse_components non-zero: {sparse_components.count_nonzero()}')
    X_hat = sparse_XV @ sparse_components
    coo = coo_matrix(X_hat)
    story_ids = user_story_matrix.col_index[coo.col].values
    user_ids = user_story_matrix.row_index[coo.row].values
    context.log.info(f'recommendations: {len(story_ids)}')
    return DataFrame.from_dict({'user_id': user_ids, 'story_id': story_ids, 'relevance': coo.data})