from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np

def _standardize_data(data, col, whitten, flatten):
    if False:
        return 10
    if col is not None:
        data = data[col]
    if whitten:
        data = StandardScaler().fit_transform(data)
    return data

def get_tsne_components(data, features_col=0, labels_col=1, whitten=True, n_components=3, perplexity=20, flatten=True, for_plot=True):
    if False:
        while True:
            i = 10
    features = _standardize_data(data, features_col, whitten, flatten)
    tsne = TSNE(n_components=n_components, perplexity=perplexity)
    tsne_results = tsne.fit_transform(features)
    if for_plot:
        comps = tsne_results.tolist()
        labels = data[labels_col]
        for (i, item) in enumerate(comps):
            label = labels[i]
            if isinstance(labels, np.ndarray):
                label = label.item()
            item.extend((None, None, None, str(int(label)), label))
        return comps
    return tsne_results