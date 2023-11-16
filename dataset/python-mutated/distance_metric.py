def cosine_similarity(embedding_tensor, query_embedding):
    if False:
        while True:
            i = 10
    return f'COSINE_SIMILARITY({embedding_tensor}, {query_embedding})'

def l1_norm(embedding_tensor, query_embedding):
    if False:
        return 10
    return f'L1_NORM({embedding_tensor}-{query_embedding})'

def l2_norm(embedding_tensor, query_embedding):
    if False:
        i = 10
        return i + 15
    return f'L2_NORM({embedding_tensor}-{query_embedding})'

def linf_norm(embedding_tensor, query_embedding):
    if False:
        return 10
    return f'LINF_NORM({embedding_tensor}-{query_embedding})'

def deepmemory_distance(embedding_tensor, query_embedding):
    if False:
        for i in range(10):
            print('nop')
    return f'deepmemory_distance({embedding_tensor}, {query_embedding})'
METRIC_TO_TQL_QUERY = {'l1': l1_norm, 'l2': l2_norm, 'cos': cosine_similarity, 'max': linf_norm, 'deepmemory_distance': deepmemory_distance}

def get_tql_distance_metric(distance_metric, embedding_tensor, query_embedding):
    if False:
        i = 10
        return i + 15
    metric_fn = METRIC_TO_TQL_QUERY[distance_metric]
    return metric_fn(embedding_tensor, query_embedding)