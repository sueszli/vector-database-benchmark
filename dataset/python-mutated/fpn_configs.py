"""BiFPN/QuFPN and other FPN configs.

BiFPN is presented in the EfficientDet paper.
QuFPN is proposed in https://github.com/google/automl/pull/580
"""
import itertools
import hparams_config

def bifpn_config(min_level, max_level, weight_method):
    if False:
        while True:
            i = 10
    'A dynamic bifpn config that can adapt to different min/max levels.'
    p = hparams_config.Config()
    p.weight_method = weight_method or 'fastattn'
    num_levels = max_level - min_level + 1
    node_ids = {min_level + i: [i] for i in range(num_levels)}
    level_last_id = lambda level: node_ids[level][-1]
    level_all_ids = lambda level: node_ids[level]
    id_cnt = itertools.count(num_levels)
    p.nodes = []
    for i in range(max_level - 1, min_level - 1, -1):
        p.nodes.append({'feat_level': i, 'inputs_offsets': [level_last_id(i), level_last_id(i + 1)]})
        node_ids[i].append(next(id_cnt))
    for i in range(min_level + 1, max_level + 1):
        p.nodes.append({'feat_level': i, 'inputs_offsets': level_all_ids(i) + [level_last_id(i - 1)]})
        node_ids[i].append(next(id_cnt))
    return p

def qufpn_config(min_level, max_level, weight_method=None):
    if False:
        for i in range(10):
            print('nop')
    'A dynamic quad fpn config that can adapt to different min/max levels.'
    p = hparams_config.Config()
    p.weight_method = weight_method or 'fastattn'
    p.quad_method = 'fastattn'
    num_levels = max_level - min_level + 1
    node_ids = {min_level + i: [i] for i in range(num_levels)}
    level_last_id = lambda level: node_ids[level][-1]
    level_all_ids = lambda level: node_ids[level]
    level_first_id = lambda level: node_ids[level][0]
    id_cnt = itertools.count(num_levels)
    p.nodes = []
    for i in range(max_level - 1, min_level - 1, -1):
        p.nodes.append({'feat_level': i, 'inputs_offsets': [level_last_id(i), level_last_id(i + 1)], 'weight_method': p.weight_method})
        node_ids[i].append(next(id_cnt))
    node_ids[max_level].append(node_ids[max_level][-1])
    for i in range(min_level + 1, max_level):
        p.nodes.append({'feat_level': i, 'inputs_offsets': level_all_ids(i) + [level_last_id(i - 1)], 'weight_method': p.weight_method})
        node_ids[i].append(next(id_cnt))
    i = max_level
    p.nodes.append({'feat_level': i, 'inputs_offsets': [level_first_id(i)] + [level_last_id(i - 1)], 'weight_method': p.weight_method})
    node_ids[i].append(next(id_cnt))
    node_ids[min_level].append(node_ids[min_level][-1])
    for i in range(min_level + 1, max_level + 1, 1):
        p.nodes.append({'feat_level': i, 'inputs_offsets': [level_first_id(i), level_last_id(i - 1) if i != min_level + 1 else level_first_id(i - 1)], 'weight_method': p.weight_method})
        node_ids[i].append(next(id_cnt))
    node_ids[min_level].append(node_ids[min_level][-1])
    for i in range(max_level - 1, min_level, -1):
        p.nodes.append({'feat_level': i, 'inputs_offsets': [node_ids[i][0]] + [node_ids[i][-1]] + [level_last_id(i + 1)], 'weight_method': p.weight_method})
        node_ids[i].append(next(id_cnt))
    i = min_level
    p.nodes.append({'feat_level': i, 'inputs_offsets': [node_ids[i][0]] + [level_last_id(i + 1)], 'weight_method': p.weight_method})
    node_ids[i].append(next(id_cnt))
    node_ids[max_level].append(node_ids[max_level][-1])
    for i in range(max_level, min_level - 1, -1):
        p.nodes.append({'feat_level': i, 'inputs_offsets': [node_ids[i][2], node_ids[i][4]], 'weight_method': p.quad_method})
        node_ids[i].append(next(id_cnt))
    return p

def get_fpn_config(fpn_name, min_level, max_level, weight_method):
    if False:
        print('Hello World!')
    'Get fpn related configuration.'
    if not fpn_name:
        fpn_name = 'bifpn'
    name_to_config = {'bifpn': bifpn_config(min_level, max_level, weight_method), 'qufpn': qufpn_config(min_level, max_level, weight_method), 'bifpn_dyn': bifpn_config(min_level, max_level, weight_method)}
    return name_to_config[fpn_name]