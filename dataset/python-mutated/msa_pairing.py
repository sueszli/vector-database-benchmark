"""Pairing logic for multimer data """
import collections
from typing import Dict, Iterable, List, Sequence
import numpy as np
import pandas as pd
import scipy.linalg
from .data_ops import NumpyDict
from .residue_constants import restypes_with_x_and_gap
MSA_GAP_IDX = restypes_with_x_and_gap.index('-')
SEQUENCE_GAP_CUTOFF = 0.5
SEQUENCE_SIMILARITY_CUTOFF = 0.9
MSA_PAD_VALUES = {'msa_all_seq': MSA_GAP_IDX, 'msa_mask_all_seq': 1, 'deletion_matrix_all_seq': 0, 'deletion_matrix_int_all_seq': 0, 'msa': MSA_GAP_IDX, 'msa_mask': 1, 'deletion_matrix': 0, 'deletion_matrix_int': 0}
MSA_FEATURES = ('msa', 'msa_mask', 'deletion_matrix', 'deletion_matrix_int')
SEQ_FEATURES = ('residue_index', 'aatype', 'all_atom_positions', 'all_atom_mask', 'seq_mask', 'between_segment_residues', 'has_alt_locations', 'has_hetatoms', 'asym_id', 'entity_id', 'sym_id', 'entity_mask', 'deletion_mean', 'prediction_atom_mask', 'literature_positions', 'atom_indices_to_group_indices', 'rigid_group_default_frame', 'num_sym')
TEMPLATE_FEATURES = ('template_aatype', 'template_all_atom_positions', 'template_all_atom_mask')
CHAIN_FEATURES = ('num_alignments', 'seq_length')

def create_paired_features(chains: Iterable[NumpyDict]) -> List[NumpyDict]:
    if False:
        i = 10
        return i + 15
    'Returns the original chains with paired NUM_SEQ features.\n\n    Args:\n        chains:    A list of feature dictionaries for each chain.\n\n    Returns:\n        A list of feature dictionaries with sequence features including only\n        rows to be paired.\n    '
    chains = list(chains)
    chain_keys = chains[0].keys()
    if len(chains) < 2:
        return chains
    else:
        updated_chains = []
        paired_chains_to_paired_row_indices = pair_sequences(chains)
        paired_rows = reorder_paired_rows(paired_chains_to_paired_row_indices)
        for (chain_num, chain) in enumerate(chains):
            new_chain = {k: v for (k, v) in chain.items() if '_all_seq' not in k}
            for feature_name in chain_keys:
                if feature_name.endswith('_all_seq'):
                    feats_padded = pad_features(chain[feature_name], feature_name)
                    new_chain[feature_name] = feats_padded[paired_rows[:, chain_num]]
            new_chain['num_alignments_all_seq'] = np.asarray(len(paired_rows[:, chain_num]))
            updated_chains.append(new_chain)
        return updated_chains

def pad_features(feature: np.ndarray, feature_name: str) -> np.ndarray:
    if False:
        return 10
    "Add a 'padding' row at the end of the features list.\n\n    The padding row will be selected as a 'paired' row in the case of partial\n    alignment - for the chain that doesn't have paired alignment.\n\n    Args:\n        feature: The feature to be padded.\n        feature_name: The name of the feature to be padded.\n\n    Returns:\n        The feature with an additional padding row.\n    "
    assert feature.dtype != np.dtype(np.str_)
    if feature_name in ('msa_all_seq', 'msa_mask_all_seq', 'deletion_matrix_all_seq', 'deletion_matrix_int_all_seq'):
        num_res = feature.shape[1]
        padding = MSA_PAD_VALUES[feature_name] * np.ones([1, num_res], feature.dtype)
    elif feature_name == 'msa_species_identifiers_all_seq':
        padding = [b'']
    else:
        return feature
    feats_padded = np.concatenate([feature, padding], axis=0)
    return feats_padded

def _make_msa_df(chain_features: NumpyDict) -> pd.DataFrame:
    if False:
        print('Hello World!')
    'Makes dataframe with msa features needed for msa pairing.'
    chain_msa = chain_features['msa_all_seq']
    query_seq = chain_msa[0]
    per_seq_similarity = np.sum(query_seq[None] == chain_msa, axis=-1) / float(len(query_seq))
    per_seq_gap = np.sum(chain_msa == 21, axis=-1) / float(len(query_seq))
    msa_df = pd.DataFrame({'msa_species_identifiers': chain_features['msa_species_identifiers_all_seq'], 'msa_row': np.arange(len(chain_features['msa_species_identifiers_all_seq'])), 'msa_similarity': per_seq_similarity, 'gap': per_seq_gap})
    return msa_df

def _create_species_dict(msa_df: pd.DataFrame) -> Dict[bytes, pd.DataFrame]:
    if False:
        for i in range(10):
            print('nop')
    'Creates mapping from species to msa dataframe of that species.'
    species_lookup = {}
    for (species, species_df) in msa_df.groupby('msa_species_identifiers'):
        species_lookup[species] = species_df
    return species_lookup

def _match_rows_by_sequence_similarity(this_species_msa_dfs: List[pd.DataFrame]) -> List[List[int]]:
    if False:
        i = 10
        return i + 15
    "Finds MSA sequence pairings across chains based on sequence similarity.\n\n    Each chain's MSA sequences are first sorted by their sequence similarity to\n    their respective target sequence. The sequences are then paired, starting\n    from the sequences most similar to their target sequence.\n\n    Args:\n        this_species_msa_dfs: a list of dataframes containing MSA features for\n            sequences for a specific species.\n\n    Returns:\n     A list of lists, each containing M indices corresponding to paired MSA rows,\n     where M is the number of chains.\n    "
    all_paired_msa_rows = []
    num_seqs = [len(species_df) for species_df in this_species_msa_dfs if species_df is not None]
    take_num_seqs = np.min(num_seqs)

    def sort_by_similarity(x):
        if False:
            return 10
        return x.sort_values('msa_similarity', axis=0, ascending=False)
    for species_df in this_species_msa_dfs:
        if species_df is not None:
            species_df_sorted = sort_by_similarity(species_df)
            msa_rows = species_df_sorted.msa_row.iloc[:take_num_seqs].values
        else:
            msa_rows = [-1] * take_num_seqs
        all_paired_msa_rows.append(msa_rows)
    all_paired_msa_rows = list(np.array(all_paired_msa_rows).transpose())
    return all_paired_msa_rows

def pair_sequences(examples: List[NumpyDict]) -> Dict[int, np.ndarray]:
    if False:
        for i in range(10):
            print('nop')
    'Returns indices for paired MSA sequences across chains.'
    num_examples = len(examples)
    all_chain_species_dict = []
    common_species = set()
    for chain_features in examples:
        msa_df = _make_msa_df(chain_features)
        species_dict = _create_species_dict(msa_df)
        all_chain_species_dict.append(species_dict)
        common_species.update(set(species_dict))
    common_species = sorted(common_species)
    common_species.remove(b'')
    all_paired_msa_rows = [np.zeros(len(examples), int)]
    all_paired_msa_rows_dict = {k: [] for k in range(num_examples)}
    all_paired_msa_rows_dict[num_examples] = [np.zeros(len(examples), int)]
    for species in common_species:
        if not species:
            continue
        this_species_msa_dfs = []
        species_dfs_present = 0
        for species_dict in all_chain_species_dict:
            if species in species_dict:
                this_species_msa_dfs.append(species_dict[species])
                species_dfs_present += 1
            else:
                this_species_msa_dfs.append(None)
        if species_dfs_present <= 1:
            continue
        if np.any(np.array([len(species_df) for species_df in this_species_msa_dfs if isinstance(species_df, pd.DataFrame)]) > 600):
            continue
        paired_msa_rows = _match_rows_by_sequence_similarity(this_species_msa_dfs)
        all_paired_msa_rows.extend(paired_msa_rows)
        all_paired_msa_rows_dict[species_dfs_present].extend(paired_msa_rows)
    all_paired_msa_rows_dict = {num_examples: np.array(paired_msa_rows) for (num_examples, paired_msa_rows) in all_paired_msa_rows_dict.items()}
    return all_paired_msa_rows_dict

def reorder_paired_rows(all_paired_msa_rows_dict: Dict[int, np.ndarray]) -> np.ndarray:
    if False:
        return 10
    'Creates a list of indices of paired MSA rows across chains.\n\n    Args:\n        all_paired_msa_rows_dict: a mapping from the number of paired chains to the\n            paired indices.\n\n    Returns:\n        a list of lists, each containing indices of paired MSA rows across chains.\n        The paired-index lists are ordered by:\n            1) the number of chains in the paired alignment, i.e, all-chain pairings\n                 will come first.\n            2) e-values\n    '
    all_paired_msa_rows = []
    for num_pairings in sorted(all_paired_msa_rows_dict, reverse=True):
        paired_rows = all_paired_msa_rows_dict[num_pairings]
        paired_rows_product = np.abs(np.array([np.prod(rows.astype(np.float64)) for rows in paired_rows]))
        paired_rows_sort_index = np.argsort(paired_rows_product)
        all_paired_msa_rows.extend(paired_rows[paired_rows_sort_index])
    return np.array(all_paired_msa_rows)

def block_diag(*arrs: np.ndarray, pad_value: float=0.0) -> np.ndarray:
    if False:
        return 10
    'Like scipy.linalg.block_diag but with an optional padding value.'
    ones_arrs = [np.ones_like(x) for x in arrs]
    off_diag_mask = 1 - scipy.linalg.block_diag(*ones_arrs)
    diag = scipy.linalg.block_diag(*arrs)
    diag += (off_diag_mask * pad_value).astype(diag.dtype)
    return diag

def _correct_post_merged_feats(np_example: NumpyDict, np_chains_list: Sequence[NumpyDict], pair_msa_sequences: bool) -> NumpyDict:
    if False:
        for i in range(10):
            print('nop')
    'Adds features that need to be computed/recomputed post merging.'
    np_example['seq_length'] = np.asarray(np_example['aatype'].shape[0], dtype=np.int32)
    np_example['num_alignments'] = np.asarray(np_example['msa'].shape[0], dtype=np.int32)
    if not pair_msa_sequences:
        cluster_bias_masks = []
        for chain in np_chains_list:
            mask = np.zeros(chain['msa'].shape[0])
            mask[0] = 1
            cluster_bias_masks.append(mask)
        np_example['cluster_bias_mask'] = np.concatenate(cluster_bias_masks)
        msa_masks = [np.ones(x['msa'].shape, dtype=np.int8) for x in np_chains_list]
        np_example['bert_mask'] = block_diag(*msa_masks, pad_value=0)
    else:
        np_example['cluster_bias_mask'] = np.zeros(np_example['msa'].shape[0])
        np_example['cluster_bias_mask'][0] = 1
        msa_masks = [np.ones(x['msa'].shape, dtype=np.int8) for x in np_chains_list]
        msa_masks_all_seq = [np.ones(x['msa_all_seq'].shape, dtype=np.int8) for x in np_chains_list]
        msa_mask_block_diag = block_diag(*msa_masks, pad_value=0)
        msa_mask_all_seq = np.concatenate(msa_masks_all_seq, axis=1)
        np_example['bert_mask'] = np.concatenate([msa_mask_all_seq, msa_mask_block_diag], axis=0)
    return np_example

def _pad_templates(chains: Sequence[NumpyDict], max_templates: int) -> Sequence[NumpyDict]:
    if False:
        for i in range(10):
            print('nop')
    'For each chain pad the number of templates to a fixed size.\n\n    Args:\n        chains: A list of protein chains.\n        max_templates: Each chain will be padded to have this many templates.\n\n    Returns:\n        The list of chains, updated to have template features padded to\n        max_templates.\n    '
    for chain in chains:
        for (k, v) in chain.items():
            if k in TEMPLATE_FEATURES:
                padding = np.zeros_like(v.shape)
                padding[0] = max_templates - v.shape[0]
                padding = [(0, p) for p in padding]
                chain[k] = np.pad(v, padding, mode='constant')
    return chains

def _merge_features_from_multiple_chains(chains: Sequence[NumpyDict], pair_msa_sequences: bool) -> NumpyDict:
    if False:
        i = 10
        return i + 15
    'Merge features from multiple chains.\n\n    Args:\n        chains: A list of feature dictionaries that we want to merge.\n        pair_msa_sequences: Whether to concatenate MSA features along the\n            num_res dimension (if True), or to block diagonalize them (if False).\n\n    Returns:\n        A feature dictionary for the merged example.\n    '
    merged_example = {}
    for feature_name in chains[0]:
        feats = [x[feature_name] for x in chains]
        feature_name_split = feature_name.split('_all_seq')[0]
        if feature_name_split in MSA_FEATURES:
            if pair_msa_sequences or '_all_seq' in feature_name:
                merged_example[feature_name] = np.concatenate(feats, axis=1)
                if feature_name_split == 'msa':
                    merged_example['msa_chains_all_seq'] = np.ones(merged_example[feature_name].shape[0]).reshape(-1, 1)
            else:
                merged_example[feature_name] = block_diag(*feats, pad_value=MSA_PAD_VALUES[feature_name])
                if feature_name_split == 'msa':
                    msa_chains = []
                    for (i, feat) in enumerate(feats):
                        cur_shape = feat.shape[0]
                        vals = np.ones(cur_shape) * (i + 2)
                        msa_chains.append(vals)
                    merged_example['msa_chains'] = np.concatenate(msa_chains).reshape(-1, 1)
        elif feature_name_split in SEQ_FEATURES:
            merged_example[feature_name] = np.concatenate(feats, axis=0)
        elif feature_name_split in TEMPLATE_FEATURES:
            merged_example[feature_name] = np.concatenate(feats, axis=1)
        elif feature_name_split in CHAIN_FEATURES:
            merged_example[feature_name] = np.sum(feats).astype(np.int32)
        else:
            merged_example[feature_name] = feats[0]
    return merged_example

def _merge_homomers_dense_msa(chains: Iterable[NumpyDict]) -> Sequence[NumpyDict]:
    if False:
        print('Hello World!')
    'Merge all identical chains, making the resulting MSA dense.\n\n    Args:\n        chains: An iterable of features for each chain.\n\n    Returns:\n        A list of feature dictionaries.    All features with the same entity_id\n        will be merged - MSA features will be concatenated along the num_res\n        dimension - making them dense.\n    '
    entity_chains = collections.defaultdict(list)
    for chain in chains:
        entity_id = chain['entity_id'][0]
        entity_chains[entity_id].append(chain)
    grouped_chains = []
    for entity_id in sorted(entity_chains):
        chains = entity_chains[entity_id]
        grouped_chains.append(chains)
    chains = [_merge_features_from_multiple_chains(chains, pair_msa_sequences=True) for chains in grouped_chains]
    return chains

def _concatenate_paired_and_unpaired_features(example: NumpyDict) -> NumpyDict:
    if False:
        print('Hello World!')
    'Merges paired and block-diagonalised features.'
    features = MSA_FEATURES + ('msa_chains',)
    for feature_name in features:
        if feature_name in example:
            feat = example[feature_name]
            feat_all_seq = example[feature_name + '_all_seq']
            try:
                merged_feat = np.concatenate([feat_all_seq, feat], axis=0)
            except Exception as ex:
                raise Exception('concat failed.', feature_name, feat_all_seq.shape, feat.shape, ex.__class__, ex)
            example[feature_name] = merged_feat
    example['num_alignments'] = np.array(example['msa'].shape[0], dtype=np.int32)
    return example

def merge_chain_features(np_chains_list: List[NumpyDict], pair_msa_sequences: bool, max_templates: int) -> NumpyDict:
    if False:
        i = 10
        return i + 15
    'Merges features for multiple chains to single FeatureDict.\n\n    Args:\n        np_chains_list: List of FeatureDicts for each chain.\n        pair_msa_sequences: Whether to merge paired MSAs.\n        max_templates: The maximum number of templates to include.\n\n    Returns:\n        Single FeatureDict for entire complex.\n    '
    np_chains_list = _pad_templates(np_chains_list, max_templates=max_templates)
    np_chains_list = _merge_homomers_dense_msa(np_chains_list)
    np_example = _merge_features_from_multiple_chains(np_chains_list, pair_msa_sequences=False)
    if pair_msa_sequences:
        np_example = _concatenate_paired_and_unpaired_features(np_example)
    np_example = _correct_post_merged_feats(np_example=np_example, np_chains_list=np_chains_list, pair_msa_sequences=pair_msa_sequences)
    return np_example

def deduplicate_unpaired_sequences(np_chains: List[NumpyDict]) -> List[NumpyDict]:
    if False:
        for i in range(10):
            print('nop')
    'Removes unpaired sequences which duplicate a paired sequence.'
    feature_names = np_chains[0].keys()
    msa_features = MSA_FEATURES
    cache_msa_features = {}
    for chain in np_chains:
        entity_id = int(chain['entity_id'][0])
        if entity_id not in cache_msa_features:
            sequence_set = set((s.tobytes() for s in chain['msa_all_seq']))
            keep_rows = []
            for (row_num, seq) in enumerate(chain['msa']):
                if seq.tobytes() not in sequence_set:
                    keep_rows.append(row_num)
            new_msa_features = {}
            for feature_name in feature_names:
                if feature_name in msa_features:
                    if keep_rows:
                        new_msa_features[feature_name] = chain[feature_name][keep_rows]
                    else:
                        new_shape = list(chain[feature_name].shape)
                        new_shape[0] = 0
                        new_msa_features[feature_name] = np.zeros(new_shape, dtype=chain[feature_name].dtype)
            cache_msa_features[entity_id] = new_msa_features
        for feature_name in cache_msa_features[entity_id]:
            chain[feature_name] = cache_msa_features[entity_id][feature_name]
        chain['num_alignments'] = np.array(chain['msa'].shape[0], dtype=np.int32)
    return np_chains